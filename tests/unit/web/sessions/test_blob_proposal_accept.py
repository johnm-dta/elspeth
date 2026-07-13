from __future__ import annotations

import asyncio
import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import structlog
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretBytes
from sqlalchemy.pool import StaticPool

from elspeth.contracts.hashing import stable_hash
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


class _ExecutionServiceFake:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def get_session_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def cleanup_session_lock(self, session_id: str) -> None:
        self._locks.pop(session_id, None)


def _make_app(tmp_path: Path, user_id: str = "alice") -> tuple[FastAPI, SessionServiceImpl]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    telemetry = build_sessions_telemetry()
    service = SessionServiceImpl(
        engine,
        telemetry=telemetry,
        log=structlog.get_logger("test"),
    )

    app = FastAPI()
    identity = UserIdentity(user_id=user_id, username=user_id)

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user
    app.state.session_service = service
    app.state.session_engine = engine
    app.state.sessions_telemetry = telemetry
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=SecretBytes(b"\x00" * 32),
    )
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    app.state.catalog_service = catalog
    app.state.operator_profile_registry = MagicMock(spec=OperatorProfileRegistry)
    app.state.plugin_snapshot_factory = lambda _user: snapshot
    app.state.composer_service = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.execution_service = _ExecutionServiceFake()
    app.state.composer_progress_registry = ComposerProgressRegistry()
    app.state.scoped_secret_resolver = None
    app.include_router(create_session_router())
    return app, service


def _patch_route_execute_tool(monkeypatch: pytest.MonkeyPatch, wrapper_factory) -> None:
    for module_name in (
        "elspeth.web.sessions.routes.composer",
        "elspeth.web.sessions.routes.composer.proposals",
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        original = getattr(module, "execute_tool", None)
        if original is None:
            continue
        monkeypatch.setattr(module, "execute_tool", wrapper_factory(original))
        return
    raise AssertionError("could not locate composer proposal route execute_tool binding")


def test_accept_update_blob_proposal_commits_without_composition_state_delta(tmp_path: Path) -> None:
    app, service = _make_app(tmp_path)
    blob_service = BlobServiceImpl(service._engine, tmp_path)
    client = TestClient(app)
    session = asyncio.run(service.create_session("alice", "Blob approval", "local"))
    session_id = session.id
    state_record = asyncio.run(
        service.save_composition_state(
            session_id,
            CompositionStateData(metadata_={"name": "Blob approval", "description": ""}, is_valid=True),
            provenance="session_seed",
        )
    )
    user_message = asyncio.run(
        service.add_message(
            session_id,
            "user",
            "Please update the report blob with the approved text.",
            writer_principal="route_user_message",
        )
    )
    blob = asyncio.run(
        blob_service.create_blob(
            session_id,
            filename="report.txt",
            content=b"original content",
            mime_type="text/plain",
            created_by="user",
        )
    )
    arguments = {"blob_id": str(blob.id), "content": "approved content"}
    proposal = asyncio.run(
        service.create_composition_proposal(
            session_id=session_id,
            tool_call_id="call_update_blob",
            tool_name="update_blob",
            summary="Update the report blob.",
            rationale="Requested by the current composer turn.",
            affects=("blob",),
            arguments_json=arguments,
            arguments_redacted_json={"blob_id": str(blob.id), "content": "<redacted>"},
            base_state_id=state_record.id,
            actor="composer-web:user:alice",
            user_message_id=user_message.id,
            composer_model_identifier="openai/gpt-5-mini",
            composer_model_version="gpt-5-mini-2026-05-01",
            composer_provider="openai",
            composer_skill_hash="sha256:composer-skill",
            tool_arguments_hash=stable_hash(arguments),
        )
    )

    response = client.post(f"/api/sessions/{session_id}/proposals/{proposal.id}/accept")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "committed"
    assert body["committed_state_id"] == str(state_record.id)
    assert Path(blob.storage_path).read_text(encoding="utf-8") == "approved content"
    persisted = asyncio.run(service.get_current_state(session_id))
    assert persisted is not None
    assert persisted.id == state_record.id


def test_accept_delete_blob_proposal_commits_without_composition_state_delta(tmp_path: Path) -> None:
    app, service = _make_app(tmp_path)
    blob_service = BlobServiceImpl(service._engine, tmp_path)
    client = TestClient(app)
    session = asyncio.run(service.create_session("alice", "Blob deletion approval", "local"))
    session_id = session.id
    state_record = asyncio.run(
        service.save_composition_state(
            session_id,
            CompositionStateData(metadata_={"name": "Blob deletion approval", "description": ""}, is_valid=True),
            provenance="session_seed",
        )
    )
    blob = asyncio.run(
        blob_service.create_blob(
            session_id,
            filename="obsolete.txt",
            content=b"obsolete content",
            mime_type="text/plain",
            created_by="user",
        )
    )
    arguments = {"blob_id": str(blob.id)}
    proposal = asyncio.run(
        service.create_composition_proposal(
            session_id=session_id,
            tool_call_id="call_delete_blob",
            tool_name="delete_blob",
            summary="Delete the obsolete blob.",
            rationale="Requested by the current composer turn.",
            affects=("blob",),
            arguments_json=arguments,
            arguments_redacted_json=arguments,
            base_state_id=state_record.id,
            actor="composer-web:user:alice",
            tool_arguments_hash=stable_hash(arguments),
            user_message_id=None,
            composer_model_identifier="openai/gpt-5-mini",
            composer_model_version="gpt-5-mini-2026-05-01",
            composer_provider="openai",
            composer_skill_hash="sha256:composer-skill",
        )
    )

    response = client.post(f"/api/sessions/{session_id}/proposals/{proposal.id}/accept")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "committed"
    assert body["committed_state_id"] == str(state_record.id)
    assert not Path(blob.storage_path).exists()
    persisted = asyncio.run(service.get_current_state(session_id))
    assert persisted is not None
    assert persisted.id == state_record.id


@pytest.mark.asyncio
async def test_accept_update_blob_proposal_serializes_against_reject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, service = _make_app(tmp_path)
    blob_service = BlobServiceImpl(service._engine, tmp_path)
    session = await service.create_session("alice", "Blob approval race", "local")
    session_id = session.id
    state_record = await service.save_composition_state(
        session_id,
        CompositionStateData(metadata_={"name": "Blob approval race", "description": ""}, is_valid=True),
        provenance="session_seed",
    )
    user_message = await service.add_message(
        session_id,
        "user",
        "Please update the report blob with the approved text.",
        writer_principal="route_user_message",
    )
    blob = await blob_service.create_blob(
        session_id,
        filename="report.txt",
        content=b"original content",
        mime_type="text/plain",
        created_by="user",
    )
    arguments = {"blob_id": str(blob.id), "content": "approved content"}
    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_update_blob_race",
        tool_name="update_blob",
        summary="Update the report blob.",
        rationale="Requested by the current composer turn.",
        affects=("blob",),
        arguments_json=arguments,
        arguments_redacted_json={"blob_id": str(blob.id), "content": "<redacted>"},
        base_state_id=state_record.id,
        actor="composer-web:user:alice",
        user_message_id=user_message.id,
        composer_model_identifier="openai/gpt-5-mini",
        composer_model_version="gpt-5-mini-2026-05-01",
        composer_provider="openai",
        composer_skill_hash="sha256:composer-skill",
        tool_arguments_hash=stable_hash(arguments),
    )
    entered_tool = threading.Event()
    release_tool = threading.Event()

    def wrapper_factory(original):
        def gated_execute_tool(*args, **kwargs):
            entered_tool.set()
            if not release_tool.wait(timeout=5):
                raise TimeoutError("test timed out waiting to release execute_tool")
            return original(*args, **kwargs)

        return gated_execute_tool

    _patch_route_execute_tool(monkeypatch, wrapper_factory)

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            accept_task = asyncio.create_task(client.post(f"/api/sessions/{session_id}/proposals/{proposal.id}/accept"))
            assert await asyncio.to_thread(entered_tool.wait, 5)

            reject_task = asyncio.create_task(
                client.post(
                    f"/api/sessions/{session_id}/proposals/{proposal.id}/reject",
                    json={"reason": "operator changed mind"},
                )
            )
            await asyncio.sleep(0.05)
            assert not reject_task.done()

            release_tool.set()
            accept_response, reject_response = await asyncio.gather(accept_task, reject_task)
    finally:
        release_tool.set()

    assert accept_response.status_code == 200
    assert reject_response.status_code == 409
    assert Path(blob.storage_path).read_text(encoding="utf-8") == "approved content"
    proposals = await service.list_composition_proposals(session_id)
    assert [item.status for item in proposals] == ["committed"]
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == [
        "proposal.created",
        "proposal.accepted",
    ]


@pytest.mark.asyncio
async def test_cancelled_accept_update_blob_proposal_still_terminalizes_before_reject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, service = _make_app(tmp_path)
    blob_service = BlobServiceImpl(service._engine, tmp_path)
    session = await service.create_session("alice", "Blob approval cancellation", "local")
    session_id = session.id
    state_record = await service.save_composition_state(
        session_id,
        CompositionStateData(metadata_={"name": "Blob approval cancellation", "description": ""}, is_valid=True),
        provenance="session_seed",
    )
    user_message = await service.add_message(
        session_id,
        "user",
        "Please update the report blob with the approved text.",
        writer_principal="route_user_message",
    )
    blob = await blob_service.create_blob(
        session_id,
        filename="report.txt",
        content=b"original content",
        mime_type="text/plain",
        created_by="user",
    )
    arguments = {"blob_id": str(blob.id), "content": "approved content"}
    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_update_blob_cancel",
        tool_name="update_blob",
        summary="Update the report blob.",
        rationale="Requested by the current composer turn.",
        affects=("blob",),
        arguments_json=arguments,
        arguments_redacted_json={"blob_id": str(blob.id), "content": "<redacted>"},
        base_state_id=state_record.id,
        actor="composer-web:user:alice",
        user_message_id=user_message.id,
        composer_model_identifier="openai/gpt-5-mini",
        composer_model_version="gpt-5-mini-2026-05-01",
        composer_provider="openai",
        composer_skill_hash="sha256:composer-skill",
        tool_arguments_hash=stable_hash(arguments),
    )
    entered_tool = threading.Event()
    release_tool = threading.Event()

    def wrapper_factory(original):
        def gated_execute_tool(*args, **kwargs):
            entered_tool.set()
            if not release_tool.wait(timeout=5):
                raise TimeoutError("test timed out waiting to release execute_tool")
            return original(*args, **kwargs)

        return gated_execute_tool

    _patch_route_execute_tool(monkeypatch, wrapper_factory)

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            accept_task = asyncio.create_task(client.post(f"/api/sessions/{session_id}/proposals/{proposal.id}/accept"))
            assert await asyncio.to_thread(entered_tool.wait, 5)
            accept_task.cancel()

            reject_task = asyncio.create_task(
                client.post(
                    f"/api/sessions/{session_id}/proposals/{proposal.id}/reject",
                    json={"reason": "operator changed mind"},
                )
            )
            await asyncio.sleep(0.05)
            assert not reject_task.done()

            release_tool.set()
            with pytest.raises(asyncio.CancelledError):
                await accept_task
            reject_response = await reject_task
    finally:
        release_tool.set()

    assert reject_response.status_code == 409
    assert Path(blob.storage_path).read_text(encoding="utf-8") == "approved content"
    proposals = await service.list_composition_proposals(session_id)
    assert [item.status for item in proposals] == ["committed"]
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == [
        "proposal.created",
        "proposal.accepted",
    ]
