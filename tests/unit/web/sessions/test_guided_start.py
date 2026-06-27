"""POST /api/sessions/{session_id}/guided/start — idempotent profile-seeded guided entry (P6, §4.3, D16)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from fastapi import FastAPI
from sqlalchemy.pool import StaticPool

from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _make_app(tmp_path, user_id="alice"):
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )
    app = FastAPI()
    identity = UserIdentity(user_id=user_id, username=user_id)

    async def mock_user():
        return identity

    app.dependency_overrides[get_current_user] = mock_user
    app.state.session_service = service
    app.state.session_engine = engine
    catalog = MagicMock(spec=["list_sources", "list_transforms", "list_sinks", "get_schema"])
    catalog.list_sources.return_value = [
        PluginSummary(
            name="inline_blob",
            description="Inline blob source",
            plugin_type="source",
            config_fields=[],
        ),
    ]
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="inline_blob",
        plugin_type="source",
        description="Inline blob source",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    app.state.catalog_service = catalog
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.payload_store = FilesystemPayloadStore(app.state.settings.get_payload_store_path())
    app.state.blob_service = MagicMock()
    app.state.blob_service.list_blobs = AsyncMock(return_value=[])
    app.state.blob_service.create_blob = AsyncMock()
    app.state.blob_service.get_blob = AsyncMock()
    app.state.composer_service = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.composer_progress_registry = ComposerProgressRegistry()
    app.include_router(create_session_router())
    return app, service


@pytest.mark.asyncio
async def test_guided_start_seeds_tutorial_profile_and_persists(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Wire carries the tutorial profile (advisor_checkpoints OFF — matches live
    # guided; bookends on).
    assert body["guided_session"]["profile"] is not None
    assert body["guided_session"]["profile"]["advisor_checkpoints"] is False
    assert body["guided_session"]["profile"]["bookends"] is True

    get_resp = client.get(f"/api/sessions/{session.id}/guided")
    assert get_resp.status_code == 200
    assert get_resp.json()["guided_session"]["profile"]["advisor_checkpoints"] is False


@pytest.mark.asyncio
async def test_guided_start_persists_profile_without_materializing_topology(tmp_path) -> None:
    """Tutorial start persists the profile-seeded GuidedSession (D16 decision: D).

    The client only sends {"profile": "tutorial"}. ``start`` constructs the
    SERVER-owned ``TUTORIAL_PROFILE`` and persists it on the GuidedSession; it
    does NOT fabricate any source/topology into the CompositionState — no chat
    turn is injected. The real pipeline is wizard/recipe-built downstream.
    """
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial"},
    )
    assert resp.status_code == 200
    body = resp.json()

    # The tutorial profile is on the wire (populated subset).
    assert body["guided_session"]["profile"] is not None
    assert body["guided_session"]["profile"]["advisor_checkpoints"] is False

    # D contract: start does NOT materialize a chat turn or any topology.
    # Empty CompositionState on the wire: sources={} (mapping), nodes/edges/
    # outputs=[] (lists).
    assert body["guided_session"]["chat_history"] == []
    assert body["guided_session"]["chat_turn_seq"] == 0
    state = body["composition_state"]
    assert state["sources"] == {}
    assert state["nodes"] == []
    assert state["edges"] == []
    assert state["outputs"] == []

    # The profile is persisted: GET reflects the populated tutorial profile and
    # the same empty (un-materialized) composition state — and still no leak.
    get_resp = client.get(f"/api/sessions/{session.id}/guided")
    assert get_resp.status_code == 200
    get_body = get_resp.json()
    assert get_body["guided_session"]["profile"]["advisor_checkpoints"] is False
    assert get_body["guided_session"]["chat_history"] == []

    # Re-read the persisted record: empty source/topology (tuples on the record).
    persisted = await service.get_current_state(session.id)
    assert persisted is not None
    assert dict(persisted.sources) == {}
    assert tuple(persisted.nodes) == ()
    assert tuple(persisted.edges) == ()
    assert tuple(persisted.outputs) == ()


@pytest.mark.asyncio
async def test_guided_start_live_profile_is_empty(tmp_path) -> None:
    """The live profile maps to EMPTY_PROFILE: wire profile is None, state empty."""
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "live"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # live == empty profile: wire profile is None and no chat/topology exists.
    assert body["guided_session"]["profile"] is None
    assert body["guided_session"]["chat_history"] == []
    assert body["guided_session"]["chat_turn_seq"] == 0
    assert body["composition_state"]["sources"] == {}
    assert body["composition_state"]["nodes"] == []


@pytest.mark.asyncio
async def test_guided_start_is_idempotent(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    first = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial"},
    )
    assert first.status_code == 200

    second = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "live"},
    )
    assert second.status_code == 200
    assert second.json()["guided_session"]["profile"] is not None
    assert second.json()["guided_session"]["profile"]["advisor_checkpoints"] is False

    from sqlalchemy import text

    with service._engine.connect() as conn:
        versions = conn.execute(
            text("SELECT COUNT(*) FROM composition_states WHERE session_id = :sid"),
            {"sid": str(session.id)},
        ).scalar()
    assert versions == 1


@pytest.mark.asyncio
async def test_guided_start_rejects_existing_freeform_state_without_guided_session(tmp_path) -> None:
    """Do not silently convert or overwrite a freeform composition state."""
    from elspeth.web.sessions.protocol import CompositionStateData

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "Freeform draft", "local")
    # A complete freeform SourceSpec so the persisted record round-trips
    # through _state_from_record (the route reconstructs the state to check
    # for a guided_session). A real compose path persists the full source
    # shape (plugin/options/on_success/on_validation_failure) plus
    # PipelineMetadata; supply both so the Tier-1 reconstruction guards pass
    # and the route reaches its 409-on-freeform branch.
    freeform_source = {
        "draft": {
            "plugin": "csv",
            "options": {"path": "draft.csv"},
            "on_success": "sink",
            "on_validation_failure": "halt",
        }
    }
    existing = await service.save_composition_state(
        session.id,
        CompositionStateData(
            sources=freeform_source,
            nodes={},
            edges={},
            outputs={},
            metadata_={"name": "Freeform draft", "description": ""},
            composer_meta=None,
        ),
        provenance="post_compose",
    )

    resp = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial"},
    )
    assert resp.status_code == 409
    assert "existing freeform composition state" in resp.json()["detail"]

    persisted = await service.get_current_state(session.id)
    assert persisted is not None
    assert persisted.id == existing.id
    assert persisted.sources == freeform_source
    assert persisted.composer_meta is None or "guided_session" not in persisted.composer_meta


@pytest.mark.asyncio
async def test_guided_start_rejects_unknown_profile_kind(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "superuser"},
    )
    assert resp.status_code == 400
    assert "profile" in resp.json()["detail"].lower()
    assert "superuser" not in resp.json()["detail"]


@pytest.mark.asyncio
async def test_guided_start_rejects_client_supplied_profile_object_without_echo(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={
            "profile": {
                "kind": "tutorial",
                "injected": {"sources": {"evil": "client-owned"}},
                "advisor_checkpoints": False,
            },
        },
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "profile" in detail.lower()
    assert "injected" not in detail
    assert "evil" not in detail


@pytest.mark.asyncio
async def test_guided_start_unowned_session_404(tmp_path) -> None:
    app, _service = _make_app(tmp_path, user_id="alice")
    client = TestClient(app)
    resp = client.post(
        f"/api/sessions/{uuid.uuid4()}/guided/start",
        json={"profile": "tutorial"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_guided_respond_stale_step_index_409(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
    client.get(f"/api/sessions/{session.id}/guided")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/respond",
        json={"step_index": "step_3_transforms", "chosen": ["csv"]},
    )
    assert resp.status_code == 409
    assert "step_index" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_guided_respond_unknown_step_index_400(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
    client.get(f"/api/sessions/{session.id}/guided")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/respond",
        json={"step_index": "step_99_bogus", "chosen": ["csv"]},
    )
    assert resp.status_code == 400
    assert "step_index" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_guided_respond_success_preserves_tutorial_profile(tmp_path) -> None:
    """A normal respond response still carries the persisted tutorial profile."""
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
    client.get(f"/api/sessions/{session.id}/guided")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/respond",
        json={"step_index": "step_1_source", "chosen": ["inline_blob"]},
    )
    assert resp.status_code == 200
    profile = resp.json()["guided_session"]["profile"]
    assert profile is not None
    assert profile["advisor_checkpoints"] is False
    assert profile["bookends"] is True

    get_resp = client.get(f"/api/sessions/{session.id}/guided")
    assert get_resp.status_code == 200
    assert get_resp.json()["guided_session"]["profile"]["advisor_checkpoints"] is False
