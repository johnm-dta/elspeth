"""P3 Task 3 — confirm_wiring turn shows State-C or State-B shield wording.

Three tests cover the B-vs-C refinement at the route boundary:

- no-key (State C):  POST advance path returns PROMPT_SHIELD_WARNING_DRAFT.
- with-key (State B): POST advance path returns PROMPT_SHIELD_AVAILABLE_DRAFT.
- with-key GET re-render: GET /guided also returns PROMPT_SHIELD_AVAILABLE_DRAFT.

The fixture design mirrors ``surfacer_client`` from
``test_guided_commit_surfaces_reviews.py`` (W2 blast-radius rule: dedicated
fixture, do NOT mutate the base ``composer_test_client``).  The only
difference between the two fixtures below is ``app.state.scoped_secret_resolver``.
"""

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
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.routes import create_blobs_router
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.interpretation_state import PROMPT_SHIELD_AVAILABLE_DRAFT, PROMPT_SHIELD_WARNING_DRAFT
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.plugin_policy.availability import build_plugin_snapshot
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.profiles import LocalRequirementResult, OperatorProfileRegistry, RuntimeWebPluginConfig
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# ---------------------------------------------------------------------------
# Stub resolver
# ---------------------------------------------------------------------------


class _ShieldKeyResolver:
    """Stub scoped_secret_resolver that says AZURE_CONTENT_SAFETY_KEY is present."""

    def has_ref(self, user_id: str, name: str) -> bool:
        return name == "AZURE_CONTENT_SAFETY_KEY"


# ---------------------------------------------------------------------------
# Shared app builder
# ---------------------------------------------------------------------------


def _make_guided_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scoped_secret_resolver: object,
) -> FastAPI:
    """Full-stack FastAPI app with a real ComposerServiceImpl."""
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
        log=structlog.get_logger("test.guided.shield"),
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
        plugin_allowlist=("transform:passthrough", "transform:azure_prompt_shield"),
        llm_profiles={
            "task-role": {
                "provider": "bedrock",
                "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            }
        },
    )
    catalog = create_catalog_service()
    composer_service = ComposerServiceImpl.for_trained_operator(
        catalog=catalog,
        settings=settings,
        sessions_service=session_service,
        session_engine=engine,
    )

    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = _mock_user

    app.state.session_service = session_service
    app.state.session_engine = engine
    app.state.blob_service = blob_service
    app.state.payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    app.state.settings = settings
    app.state.composer_service = composer_service
    app.state.scoped_secret_resolver = scoped_secret_resolver
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.catalog_service = catalog
    runtime_policy = RuntimeWebPluginConfig.from_settings(settings)
    app.state.web_plugin_policy = compile_web_plugin_policy(
        registry=get_shared_plugin_manager(),
        settings=runtime_policy,
    )
    base_profiles = OperatorProfileRegistry(
        policy=app.state.web_plugin_policy,
        settings=runtime_policy,
    )

    class _FullSchemaProfiles:
        """Expose the real profile schema and availability used by guided commits."""

        def public_schema(self, plugin_id, full_schema, *, available_aliases):
            return base_profiles.public_schema(plugin_id, full_schema, available_aliases=available_aliases)

        def profile_availability(self, plugin_id, *, principal, inventory):
            return base_profiles.profile_availability(plugin_id, principal=principal, inventory=inventory)

        def lower_options(self, plugin_id, *, alias, safe_options):
            return base_profiles.lower_options(plugin_id, alias=alias, safe_options=safe_options)

        def check_local_requirements(self, plugin_id, alias):
            del plugin_id, alias
            return LocalRequirementResult(available=True)

    class _SnapshotInventory:
        def has_server_ref(self, name: str) -> bool:
            return scoped_secret_resolver is not None and scoped_secret_resolver.has_ref("alice", name)

        def has_user_ref(self, principal: str, name: str) -> bool:
            return scoped_secret_resolver is not None and scoped_secret_resolver.has_ref("alice", name)

        def has_ref(self, principal: str, name: str) -> bool:
            return scoped_secret_resolver is not None and scoped_secret_resolver.has_ref("alice", name)

    app.state.operator_profile_registry = _FullSchemaProfiles()
    app.state.plugin_snapshot_factory = lambda user: build_plugin_snapshot(
        policy=app.state.web_plugin_policy,
        catalog=catalog,
        profiles=app.state.operator_profile_registry,
        principal_scope=f"local:{user.user_id}",
        secret_inventory=_SnapshotInventory(),
        generation_key=b"guided-shield-policy-key",
    )
    app.state.composer_recorder = BufferingRecorder()
    app.state.composer_progress_registry = ComposerProgressRegistry()

    router = create_session_router()
    app.include_router(router)

    blobs_router = create_blobs_router()
    app.include_router(blobs_router)

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def composer_test_client_no_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Shield unavailable (scoped_secret_resolver=None) → State C wording."""
    app = _make_guided_app(tmp_path, monkeypatch, scoped_secret_resolver=None)
    yield TestClient(app)


@pytest.fixture
def composer_test_client_with_shield_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Shield available (AZURE_CONTENT_SAFETY_KEY in resolver) → State B wording."""
    app = _make_guided_app(tmp_path, monkeypatch, scoped_secret_resolver=_ShieldKeyResolver())
    yield TestClient(app)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", json={"title": "shield-state-test"})
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
    """Single-LLM-node chain proposal (mirrors test_guided_commit_surfaces_reviews)."""
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
                                                        "required_input_fields": ["text"],
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


def _drive_to_wire_turn(client: TestClient, session_id: str) -> dict:
    """Drive source → sink → transforms-chat → propose_chain → accept → confirm_wiring.

    Returns the ``confirm_wiring`` next_turn dict.
    Must be called inside a ``patch(_litellm_acompletion)`` context.

    The sink commit no longer auto-builds the transform chain (the
    sink→step_3 auto-build was removed): it advances to STEP_3_TRANSFORMS with
    ``next_turn`` null. The per-stage transforms prompt — sent via
    ``POST /guided/chat`` with ``step_index="step_3_transforms"`` — drives the
    chain build, and the SAME ``chain_solver._litellm_acompletion`` mock fires
    on that chat call, surfacing the ``propose_chain`` turn in its response.
    """
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
    # Commit the sink → advances to STEP_3_TRANSFORMS with NO auto-proposal
    # (the sink→step_3 auto-build was removed); next_turn is null here.
    body = _respond(client, session_id, chosen=["text"], custom_inputs=[])
    assert body["next_turn"] is None, body["next_turn"]
    assert body["guided_session"]["step"] == "step_3_transforms", body["guided_session"]["step"]

    # The per-stage transforms prompt drives the chain build via /guided/chat.
    # The same chain_solver._litellm_acompletion mock fires on this call, so its
    # response carries the propose_chain turn.
    chat_resp = client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={"message": "fetch each page and summarise it", "step_index": "step_3_transforms"},
    )
    assert chat_resp.status_code == 200, chat_resp.json()
    chat_body = chat_resp.json()
    assert chat_body["next_turn"]["type"] == "propose_chain", chat_body["next_turn"]

    # Accept the proposed LLM chain → produces confirm_wiring turn.
    result = _respond(client, session_id, chosen=["accept"])
    assert result["next_turn"] is not None, result
    assert result["next_turn"]["type"] == "confirm_wiring", result["next_turn"]
    return result["next_turn"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wire_turn_state_c_no_key(composer_test_client_no_key: TestClient) -> None:
    """Without AZURE_CONTENT_SAFETY_KEY, confirm_wiring warnings contain State-C wording."""
    client = composer_test_client_no_key
    session_id = _create_session(client)
    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_fake_llm_chain_response(),
    ):
        wire_turn = _drive_to_wire_turn(client, session_id)

    messages = [w["message"] for w in wire_turn["payload"]["warnings"]]
    assert messages, "Expected at least one warning; got none"
    assert any(PROMPT_SHIELD_WARNING_DRAFT in m for m in messages), (
        f"Expected State-C wording ({PROMPT_SHIELD_WARNING_DRAFT!r}); got {messages}"
    )
    assert not any(PROMPT_SHIELD_AVAILABLE_DRAFT in m for m in messages), "State-B wording must not appear when shield key is absent"


def test_wire_turn_state_b_with_key(composer_test_client_with_shield_key: TestClient) -> None:
    """With AZURE_CONTENT_SAFETY_KEY available, confirm_wiring warnings contain State-B wording."""
    client = composer_test_client_with_shield_key
    session_id = _create_session(client)
    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_fake_llm_chain_response(),
    ):
        wire_turn = _drive_to_wire_turn(client, session_id)

    messages = [w["message"] for w in wire_turn["payload"]["warnings"]]
    assert messages, "Expected at least one warning; got none"
    assert any(PROMPT_SHIELD_AVAILABLE_DRAFT in m for m in messages), (
        f"Expected State-B wording ({PROMPT_SHIELD_AVAILABLE_DRAFT!r}); got {messages}"
    )
    assert not any(PROMPT_SHIELD_WARNING_DRAFT in m for m in messages), "State-C wording must not appear when shield key is present"


def test_wire_turn_state_b_get_rerender(composer_test_client_with_shield_key: TestClient) -> None:
    """GET /guided re-render also shows State-B wording when shield key is available."""
    client = composer_test_client_with_shield_key
    session_id = _create_session(client)
    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_fake_llm_chain_response(),
    ):
        _drive_to_wire_turn(client, session_id)

    # GET after reaching STEP_4_WIRE — must also show State-B wording.
    body = _get_guided(client, session_id)
    assert body["next_turn"]["type"] == "confirm_wiring", body["next_turn"]
    messages = [w["message"] for w in body["next_turn"]["payload"]["warnings"]]
    assert messages, "Expected at least one warning on GET re-render; got none"
    assert any(PROMPT_SHIELD_AVAILABLE_DRAFT in m for m in messages), f"Expected State-B wording on GET re-render; got {messages}"
    assert not any(PROMPT_SHIELD_WARNING_DRAFT in m for m in messages), (
        "State-C wording must not appear on GET re-render when shield key is present"
    )
