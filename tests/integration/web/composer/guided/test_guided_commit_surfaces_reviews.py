"""Phase P3.2 — a guided chain-accept commit surfaces interpretation cards via the route."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
import structlog
from fastapi import FastAPI
from sqlalchemy.pool import StaticPool

from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.routes import create_blobs_router
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


@pytest.fixture
def surfacer_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient with a REAL ``ComposerServiceImpl`` wired onto app state.

    Dedicated fixture (W2 blast-radius fix): the base ``composer_test_client``
    in ``conftest.py`` leaves ``app.state.composer_service = None``, which keeps
    the guided persist-seam surfacer a no-op for the ~11 sibling tests that
    inherit it. This test must exercise the real surfacer, so it mirrors
    ``test_progressive_disclosure.py``'s ``composer_freeform_client`` to wire a
    real impl onto the same in-memory engine + catalog the route uses.

    A fake ``OPENAI_API_KEY`` env var is set so the service's boot-time
    availability check passes without a real API key; ``_litellm_acompletion``
    is patched per-test to avoid live LLM calls.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-integration-tests")

    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)

    session_service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided.surfacer"),
    )
    blob_service = BlobServiceImpl(engine, tmp_path)

    settings = WebSettings(
        data_dir=tmp_path,
        composer_model="gpt-4o-mini",
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=100,
        shareable_link_signing_key=b"\x00" * 32,
    )
    catalog = create_catalog_service()
    composer_service = ComposerServiceImpl(
        catalog=catalog,
        settings=settings,
        sessions_service=session_service,
        session_engine=engine,
    )

    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user

    app.state.session_service = session_service
    app.state.session_engine = engine
    app.state.blob_service = blob_service
    app.state.payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    app.state.settings = settings
    app.state.composer_service = composer_service
    app.state.scoped_secret_resolver = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.catalog_service = catalog
    app.state.composer_recorder = BufferingRecorder()
    app.state.composer_progress_registry = ComposerProgressRegistry()

    router = create_session_router()
    app.include_router(router)

    blobs_router = create_blobs_router()
    app.include_router(blobs_router)

    client = TestClient(app)
    yield client


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "surface-test"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _get_guided(client: TestClient, session_id: str) -> dict:
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _seed_blob(client: TestClient, session_id: str) -> tuple[str, str]:
    content = "text,note\nHello world,greeting\nGoodbye,farewell\n"
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.json()
    blob_id = resp.json()["id"]
    record = asyncio.run(client.app.state.blob_service.get_blob(UUID(blob_id)))
    return blob_id, record.storage_path


def _outputs_path(client: TestClient, filename: str) -> str:
    data_dir: Path = client.app.state.settings.data_dir
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return str(outputs_dir / filename)


def _fake_llm_chain_response() -> SimpleNamespace:
    """A LiteLLM-shaped propose_chain response carrying a single `llm` node.

    The node sets BOTH `model` and `prompt_template` so the accept commit
    auto-stages an llm_model_choice AND an llm_prompt_template requirement
    (composer/tools/_common.py:184/202), which the surfacer then materialises
    as pending interpretation events.
    """
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps(
                                    {
                                        "turn_type": "propose_chain",
                                        "payload": {
                                            "steps": [
                                                {
                                                    "plugin": "llm",
                                                    "options": {
                                                        "provider": "openrouter",
                                                        "model": "anthropic/claude-sonnet-4.6",
                                                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                                                        "prompt_template": "Summarize {{ row.text }} and return JSON.",
                                                        "schema": {"mode": "observed"},
                                                    },
                                                    "rationale": "summarise each row with an llm transform",
                                                }
                                            ],
                                            "why": "source rows need an llm summary before the sink",
                                            "blockers": [],
                                        },
                                    }
                                ),
                            )
                        )
                    ],
                )
            )
        ]
    )


def _drive_to_step_3_propose_chain(client: TestClient, session_id: str) -> str:
    """Drive source -> sink -> propose_chain (verbatim from test_step_3_e2e)."""
    _blob_id, storage_path = _seed_blob(client, session_id)
    output_path = _outputs_path(client, "out.jsonl")

    _get_guided(client, session_id)
    _respond(client, session_id, chosen=["csv"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "csv",
            "options": {"path": storage_path, "schema": {"mode": "observed"}},
            "observed_columns": ["text", "note"],
            "sample_rows": [{"text": "Hello world", "note": "greeting"}],
        },
    )
    _respond(client, session_id, chosen=["json"])
    _respond(
        client,
        session_id,
        edited_values={
            "plugin": "json",
            "options": {
                "path": output_path,
                "schema": {"mode": "observed"},
                "mode": "write",
                "collision_policy": "auto_increment",
            },
            "observed_columns": [],
            "sample_rows": [],
        },
    )
    # No classifier keyword, single output -> no recipe match. With the
    # sink->step_3 auto-build removed, committing the recipe-match step now
    # advances to step_3_transforms with NO proposal (next_turn is null);
    # the transform chain is built by the per-stage transforms prompt sent
    # via /guided/chat below.
    body = _respond(client, session_id, chosen=["text"], custom_inputs=[])
    assert body["next_turn"] is None
    assert body["guided_session"]["step"] == "step_3_transforms"

    # Drive the transform-chain build through the per-stage transforms chat.
    # The SAME chain_solver._litellm_acompletion mock (patched by the caller)
    # fires on this call and emits the propose_chain turn. Any transforms
    # intent works; the mock ignores the message.
    chat_resp = client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={"message": "fetch each page and summarise it", "step_index": "step_3_transforms"},
    )
    assert chat_resp.status_code == 200, chat_resp.json()
    chat_body = chat_resp.json()
    assert chat_body["guided_session"]["step"] == "step_3_transforms"
    assert chat_body["next_turn"]["type"] == "propose_chain"
    return session_id


def test_chain_accept_commit_surfaces_model_and_template(surfacer_client: TestClient) -> None:
    client = surfacer_client
    session_id = _create_session(client)
    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_fake_llm_chain_response(),
    ):
        _drive_to_step_3_propose_chain(client, session_id)
        # Accept the llm-node chain: handle_step_3_chain_accept commits via
        # _execute_set_pipeline; the route's persist seam then fires the surfacer.
        _respond(client, session_id, chosen=["accept"])
    resp = client.get(f"/api/sessions/{session_id}/interpretations?status=pending")
    assert resp.status_code == 200, resp.json()
    kinds = {row["kind"] for row in resp.json()["events"]}
    assert "llm_model_choice" in kinds
    assert "llm_prompt_template" in kinds
