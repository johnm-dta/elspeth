"""Integration coverage for runtime interpretation-review preflight gates."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import insert
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.web.config import WebSettings
from elspeth.web.execution.errors import UnresolvedInterpretationPlaceholderError
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    SOURCE_AUTHORING_KEY,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import sessions_table
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


def _settings(tmp_path: Path) -> WebSettings:
    (tmp_path / "runs").mkdir(exist_ok=True)
    return WebSettings(
        data_dir=tmp_path,
        landscape_url=f"sqlite:///{tmp_path}/runs/audit.db",
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )


def _session_service() -> SessionServiceImpl:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    return SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.preflight_per_class"),
    )


def _insert_session(service: SessionServiceImpl, session_id: UUID) -> None:
    with service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="interpretation preflight",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )


class _UnusedYamlGenerator:
    pass


def _execution_service(
    *,
    loop: asyncio.AbstractEventLoop,
    tmp_path: Path,
    session_service: SessionServiceImpl,
) -> ExecutionServiceImpl:
    return ExecutionServiceImpl(
        loop=loop,
        broadcaster=ProgressBroadcaster(loop),
        settings=_settings(tmp_path),
        session_service=session_service,
        yaml_generator=_UnusedYamlGenerator(),
        telemetry=build_sessions_telemetry(),
    )


def _llm_state_with_options(options: dict[str, Any]) -> CompositionStateData:
    return CompositionStateData(
        source={
            "plugin": "inline_blob",
            "on_success": "rate",
            "options": {"rows": [{"text": "hello"}]},
            "on_validation_failure": "discard",
        },
        nodes=[
            {
                "id": "rate",
                "node_type": "transform",
                "plugin": "llm",
                "input": "source",
                "on_success": "out",
                "on_error": "quarantine",
                "options": options,
            }
        ],
        edges=[],
        outputs=[
            {
                "name": "out",
                "plugin": "jsonl",
                "options": {},
                "on_write_failure": "discard",
            }
        ],
        metadata_={"name": None, "description": None},
    )


def _pending_requirement(kind: InterpretationKind, *, user_term: str) -> dict[str, Any]:
    return {
        "id": f"{kind.value}-review",
        "kind": kind.value,
        "user_term": user_term,
        "status": "pending",
        "draft": "draft under review",
        "event_id": None,
        "accepted_value": None,
        "accepted_artifact_hash": None,
        "resolved_prompt_template_hash": None,
    }


async def _seed_and_execute(
    tmp_path: Path,
    state_data: CompositionStateData,
) -> UnresolvedInterpretationPlaceholderError:
    session_service = _session_service()
    session_id = uuid4()
    _insert_session(session_service, session_id)
    await session_service.save_composition_state(
        session_id,
        state_data,
        provenance="session_seed",
    )
    execution_service = _execution_service(
        loop=asyncio.get_running_loop(),
        tmp_path=tmp_path,
        session_service=session_service,
    )
    try:
        with pytest.raises(UnresolvedInterpretationPlaceholderError) as exc_info:
            await execution_service.execute(session_id, user_id="alice")
    finally:
        await execution_service.shutdown()
    return exc_info.value


@pytest.mark.asyncio
async def test_execute_rejects_unreviewed_vague_term(tmp_path: Path) -> None:
    exc = await _seed_and_execute(
        tmp_path,
        _llm_state_with_options(
            {
                "prompt_template": "Rate how {{ interpretation: primary colour }} this page is.",
                "model": "stub-model",
            }
        ),
    )

    # This hand-built fixture skips the mutation-time auto-stager, so the LLM
    # node also surfaces the prompt-template and (because it declares a model)
    # model-choice review sites. Assert the vague_term site this test owns is
    # flagged; the sibling auto-stage sites are expected, not a regression.
    site_pairs = [(site.component_id, site.kind) for site in exc.sites]
    assert ("rate", InterpretationKind.VAGUE_TERM) in site_pairs
    assert {kind for _, kind in site_pairs} <= {
        InterpretationKind.VAGUE_TERM,
        InterpretationKind.LLM_PROMPT_TEMPLATE,
        InterpretationKind.LLM_MODEL_CHOICE,
    }
    assert "{{interpretation:primary colour}}" in str(exc)


@pytest.mark.asyncio
async def test_execute_rejects_unreviewed_invented_source(tmp_path: Path) -> None:
    exc = await _seed_and_execute(
        tmp_path,
        CompositionStateData(
            source={
                "plugin": "inline_blob",
                "on_success": "out",
                "options": {
                    "rows": [{"url": "https://example.gov.au"}],
                    SOURCE_AUTHORING_KEY: {
                        "modality": "llm_generated",
                        "content_hash": "a" * 64,
                        "review_event_id": None,
                        "resolved_kind": None,
                    },
                    INTERPRETATION_REQUIREMENTS_KEY: [
                        _pending_requirement(
                            InterpretationKind.INVENTED_SOURCE,
                            user_term="source URL list",
                        )
                    ],
                },
                "on_validation_failure": "discard",
            },
            nodes=[],
            edges=[],
            outputs=[
                {
                    "name": "out",
                    "plugin": "jsonl",
                    "options": {},
                    "on_write_failure": "discard",
                }
            ],
            metadata_={"name": None, "description": None},
        ),
    )

    assert [(site.component_id, site.kind) for site in exc.sites] == [("source", InterpretationKind.INVENTED_SOURCE)]
    assert "invented_source" in str(exc)


@pytest.mark.asyncio
async def test_execute_rejects_unreviewed_llm_prompt_template(tmp_path: Path) -> None:
    exc = await _seed_and_execute(
        tmp_path,
        _llm_state_with_options(
            {
                "prompt_template": "Summarize {{ row.text }} for an operator.",
                "model": "stub-model",
                INTERPRETATION_REQUIREMENTS_KEY: [
                    _pending_requirement(
                        InterpretationKind.LLM_PROMPT_TEMPLATE,
                        user_term="LLM prompt template",
                    )
                ],
            }
        ),
    )

    # Hand-built fixture (no auto-stager): the declared model also surfaces a
    # model-choice review site. Assert the prompt-template site this test owns
    # is flagged; the model-choice sibling is expected, not a regression.
    site_pairs = [(site.component_id, site.kind) for site in exc.sites]
    assert ("rate", InterpretationKind.LLM_PROMPT_TEMPLATE) in site_pairs
    assert {kind for _, kind in site_pairs} <= {
        InterpretationKind.LLM_PROMPT_TEMPLATE,
        InterpretationKind.LLM_MODEL_CHOICE,
    }
    assert "llm_prompt_template" in str(exc)
