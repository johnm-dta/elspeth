"""POST /api/sessions/{session_id}/guided/start — idempotent profile-seeded guided entry (P6, §4.3, D16)."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import structlog
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.pool import StaticPool

from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import composition_states_table, guided_operation_events_table, guided_operations_table
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


class _CatalogServiceFake:
    def list_sources(self):
        return [
            PluginSummary(
                name="inline_blob",
                description="Inline blob source",
                plugin_type="source",
                config_fields=[],
            ),
        ]

    def list_transforms(self):
        return []

    def list_sinks(self):
        return []

    def get_schema(self, plugin_type, name):
        assert plugin_type == "source"
        assert name == "inline_blob"
        return PluginSchemaInfo(
            name="inline_blob",
            plugin_type="source",
            description="Inline blob source",
            json_schema={"title": "Config", "properties": {}},
            knob_schema={"fields": []},
        )


class _BlobServiceFake:
    async def list_blobs(self, *args, **kwargs):
        return []

    async def create_blob(self, *args, **kwargs):
        return None

    async def get_blob(self, *args, **kwargs):
        return None


def _make_app(tmp_path, user_id="alice", database_url: str | None = None):
    if database_url is None:
        engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    else:
        engine = create_session_engine(database_url)
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
    app.state.catalog_service = _CatalogServiceFake()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(app.state.catalog_service)
    profiles = MagicMock(spec=OperatorProfileRegistry)
    profiles.public_schema.side_effect = lambda _plugin_id, schema, *, available_aliases: schema
    app.state.operator_profile_registry = profiles
    app.state.plugin_snapshot_factory = lambda _user: snapshot
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.payload_store = FilesystemPayloadStore(app.state.settings.get_payload_store_path())
    app.state.blob_service = _BlobServiceFake()
    app.state.composer_service = None
    # Guided routes resolve shield availability via app.state; a configured
    # None resolver means "shield unavailable" (a missing key is a wiring error).
    app.state.scoped_secret_resolver = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.composer_progress_registry = ComposerProgressRegistry()
    app.include_router(create_session_router())
    return app, service


def _materialize_first_turn(state, catalog, payload_store):
    from elspeth.web.sessions.routes.composer.guided import (
        _build_get_guided_turn,
        _finalize_guided_turn,
        _prepare_server_turn_occurrence,
    )

    guided = state.guided_session
    assert guided is not None
    turn = _build_get_guided_turn(state, guided, catalog=catalog)
    assert turn is not None
    turn = _finalize_guided_turn(turn, shield_available=False)
    guided, _record, _turn_type, _prepared = _prepare_server_turn_occurrence(
        guided,
        current_step=guided.step,
        turn=turn,
        payload_store=payload_store,
    )
    return replace(state, guided_session=guided)


async def _guided_turn_emitted_args(service, session_id) -> list[dict]:
    events: list[dict] = []
    for message in await service.get_messages(session_id, limit=None):
        for tool_call in message.tool_calls or ():
            invocation = tool_call.get("invocation", {})
            if invocation.get("tool_name") == "guided_turn_emitted":
                events.append(json.loads(invocation["arguments_canonical"]))
    return events


@pytest.mark.asyncio
async def test_guided_start_seeds_tutorial_profile_and_persists(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Wire carries the tutorial profile (advisor_checkpoints OFF as the demo
    # bypass; bookends on).
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
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
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
        json={"profile": "live", "intent": "Build a live pipeline", "operation_id": str(uuid.uuid4())},
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
async def test_guided_start_profile_resolves_intent_shape_before_reservation(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    missing = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "live", "operation_id": str(uuid.uuid4())},
    )
    forbidden = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "intent": "client must not own tutorial seed", "operation_id": str(uuid.uuid4())},
    )

    assert missing.status_code == 400
    assert forbidden.status_code == 400
    with service._engine.connect() as conn:
        assert conn.execute(select(guided_operations_table)).all() == []


@pytest.mark.asyncio
async def test_fresh_get_and_start_share_the_same_prospective_turn_token(tmp_path) -> None:
    from elspeth.contracts.freeze import deep_thaw
    from elspeth.web.sessions.guided_replay import load_guided_json_payload

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())

    fresh = client.get(f"/api/sessions/{session.id}/guided")
    started = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "live", "intent": "Build a live pipeline", "operation_id": operation_id},
    )
    persisted = client.get(f"/api/sessions/{session.id}/guided")
    replayed = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "live", "intent": "Build a live pipeline", "operation_id": operation_id},
    )

    assert fresh.status_code == started.status_code == persisted.status_code == replayed.status_code == 200
    assert len(fresh.json()["guided_session"]["history"]) == 1
    assert (
        fresh.json()["next_turn"]["turn_token"]
        == started.json()["next_turn"]["turn_token"]
        == persisted.json()["next_turn"]["turn_token"]
        == replayed.json()["next_turn"]["turn_token"]
    )
    payload_id = started.json()["guided_session"]["history"][-1]["payload_hash"]
    loaded = load_guided_json_payload(
        app.state.payload_store,
        payload_id=payload_id,
        purpose="turn",
    )
    assert deep_thaw(loaded.payload) == started.json()["next_turn"]["payload"]


@pytest.mark.asyncio
async def test_guided_start_is_idempotent(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    operation_id = str(uuid.uuid4())
    first = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": operation_id},
    )
    assert first.status_code == 200

    second = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": operation_id},
    )
    assert second.status_code == 200
    assert second.json() == first.json()
    assert second.json()["guided_session"]["profile"] is not None
    assert second.json()["guided_session"]["profile"]["advisor_checkpoints"] is False

    from sqlalchemy import text

    with service._engine.connect() as conn:
        versions = conn.execute(
            text("SELECT COUNT(*) FROM composition_states WHERE session_id = :sid"),
            {"sid": str(session.id)},
        ).scalar()
    assert versions == 1
    emissions = await _guided_turn_emitted_args(service, session.id)
    assert len(emissions) == 1
    assert emissions[0]["payload_hash"] == first.json()["guided_session"]["history"][-1]["payload_hash"]
    assert emissions[0]["payload_payload_id"] == emissions[0]["payload_hash"]


@pytest.mark.asyncio
async def test_guided_start_same_operation_id_rejects_different_profile(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())

    first = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": operation_id},
    )
    conflict = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "live", "operation_id": operation_id},
    )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "Operation id is already bound to a different request."


@pytest.mark.asyncio
async def test_guided_start_existing_operation_conflict_precedes_profile_semantics(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())
    first = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": operation_id},
    )
    assert first.status_code == 200

    conflict = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "superuser", "operation_id": operation_id},
    )

    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "Operation id is already bound to a different request."
    assert "superuser" not in conflict.text


@pytest.mark.asyncio
async def test_guided_start_same_operation_object_profile_is_shape_error_before_conflict(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())
    first = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": operation_id},
    )
    assert first.status_code == 200

    invalid = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={
            "profile": {"kind": "tutorial", "injected": {"admin": True}},
            "operation_id": operation_id,
        },
    )

    assert invalid.status_code == 400
    assert "injected" not in invalid.text
    assert "admin" not in invalid.text


@pytest.mark.parametrize("hostile_kind", ["huge_integer", "deep_object"])
@pytest.mark.asyncio
async def test_guided_start_hostile_profile_rejected_before_hash_or_reservation(tmp_path, hostile_kind) -> None:
    from sqlalchemy import text

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    if hostile_kind == "huge_integer":
        hostile_profile = 2**100
        forbidden = str(hostile_profile)
    else:
        hostile_profile = {"marker": "secret-hostile-marker"}
        for _ in range(40):
            hostile_profile = {"nested": hostile_profile}
        forbidden = "secret-hostile-marker"

    response = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": hostile_profile, "operation_id": str(uuid.uuid4())},
    )

    assert response.status_code == 400
    assert forbidden not in response.text
    with service._engine.connect() as conn:
        operation_count = conn.execute(
            text("SELECT COUNT(*) FROM guided_operations WHERE session_id = :session_id"),
            {"session_id": str(session.id)},
        ).scalar_one()
    assert operation_count == 0


@pytest.mark.asyncio
async def test_guided_start_surrogate_profile_is_shape_error_without_reservation(tmp_path) -> None:
    from sqlalchemy import text

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    surrogate = "\ud800"

    response = client.post(
        f"/api/sessions/{session.id}/guided/start",
        content=(f'{{"profile":"\\ud800","operation_id":"{uuid.uuid4()}"}}').encode(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert surrogate not in response.text
    assert r"\ud800" not in response.text.lower()
    with service._engine.connect() as conn:
        operation_count = conn.execute(
            text("SELECT COUNT(*) FROM guided_operations WHERE session_id = :session_id"),
            {"session_id": str(session.id)},
        ).scalar_one()
    assert operation_count == 0


@pytest.mark.asyncio
async def test_guided_start_same_operation_surrogate_profile_does_not_mutate_operation(tmp_path) -> None:
    from sqlalchemy import text

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())
    first = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": operation_id},
    )
    assert first.status_code == 200
    operation_query = text(
        "SELECT status, request_hash, attempt, response_hash, failure_code "
        "FROM guided_operations WHERE session_id = :session_id AND operation_id = :operation_id"
    )
    with service._engine.connect() as conn:
        before = conn.execute(
            operation_query,
            {"session_id": str(session.id), "operation_id": operation_id},
        ).one()

    surrogate = "\ud800"
    response = client.post(
        f"/api/sessions/{session.id}/guided/start",
        content=(f'{{"profile":"\\ud800","operation_id":"{operation_id}"}}').encode(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert surrogate not in response.text
    assert r"\ud800" not in response.text.lower()
    with service._engine.connect() as conn:
        after = conn.execute(
            operation_query,
            {"session_id": str(session.id), "operation_id": operation_id},
        ).one()
    assert after == before


@pytest.mark.asyncio
async def test_guided_start_new_operation_returns_exact_existing_guided_head(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    first = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
    )
    current = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "live", "intent": "Build a live pipeline", "operation_id": str(uuid.uuid4())},
    )

    assert first.status_code == current.status_code == 200
    assert current.json() == first.json()


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
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
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
        json={"profile": "superuser", "operation_id": str(uuid.uuid4())},
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
            "operation_id": str(uuid.uuid4()),
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
async def test_guided_start_integrity_failure_is_terminal_and_safe_to_replay(tmp_path) -> None:
    from structlog.testing import capture_logs

    from elspeth.contracts.errors import AuditIntegrityError

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())
    payload = {"profile": "tutorial", "operation_id": operation_id}

    with (
        capture_logs() as logs,
        patch.object(
            service,
            "seed_or_complete_guided_start_operation",
            side_effect=AuditIntegrityError("secret diagnostic must not escape"),
        ),
    ):
        first = client.post(f"/api/sessions/{session.id}/guided/start", json=payload)
    replay = client.post(f"/api/sessions/{session.id}/guided/start", json=payload)

    expected = {
        "detail": {
            "error_type": "guided_operation_terminal_failure",
            "failure_code": "integrity_error",
            "detail": "The operation failed an integrity check.",
        }
    }
    assert first.status_code == replay.status_code == 500
    assert first.json() == replay.json() == expected
    event = next(entry for entry in logs if entry.get("event") == "guided.operation_terminal_failure")
    assert event["exc_class"] == "AuditIntegrityError"
    assert event["site"] == "post_guided_start"
    assert "secret diagnostic" not in repr(event)


@pytest.mark.asyncio
async def test_guided_start_audit_insert_failure_rolls_back_seed_and_occurrence(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")

    with patch.object(
        service,
        "_insert_prepared_guided_audit_rows_on_connection",
        side_effect=RuntimeError("injected audit insert failure"),
    ):
        response = client.post(
            f"/api/sessions/{session.id}/guided/start",
            json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
        )

    assert response.status_code == 500
    assert await service.get_current_state(session.id) is None
    assert await service.get_messages(session.id, limit=None) == []


@pytest.mark.asyncio
async def test_guided_start_replay_uses_durable_turn_after_live_catalog_drift(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())
    payload = {"profile": "tutorial", "operation_id": operation_id}
    first = client.post(f"/api/sessions/{session.id}/guided/start", json=payload)
    assert first.status_code == 200
    with patch(
        "elspeth.web.sessions.routes.composer.guided._build_get_guided_turn",
        side_effect=AssertionError("completed replay must not consult the live catalog"),
    ):
        replay = client.post(f"/api/sessions/{session.id}/guided/start", json=payload)

    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.json()["next_turn"]["turn_token"] == first.json()["next_turn"]["turn_token"]


@pytest.mark.asyncio
async def test_guided_start_does_not_overwrite_head_changed_after_preflight(tmp_path) -> None:
    from elspeth.web.sessions.protocol import CompositionStateData

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    original_reserve = service.reserve_guided_operation
    raced_record = None

    async def reserve_after_freeform_race(**kwargs):
        nonlocal raced_record
        outcome = await original_reserve(**kwargs)
        if raced_record is None:
            raced_record = await service.save_composition_state(
                session.id,
                CompositionStateData(
                    sources={},
                    nodes={},
                    edges={},
                    outputs={},
                    metadata_={"name": "Raced freeform", "description": ""},
                    composer_meta=None,
                ),
                provenance="post_compose",
            )
        return outcome

    with patch.object(service, "reserve_guided_operation", side_effect=reserve_after_freeform_race):
        operation_id = str(uuid.uuid4())
        response = client.post(
            f"/api/sessions/{session.id}/guided/start",
            json={"profile": "tutorial", "operation_id": operation_id},
        )
    replay = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": operation_id},
    )

    assert response.status_code == replay.status_code == 409
    assert response.json() == replay.json()
    assert response.json()["detail"]["failure_code"] == "stale_conflict"
    persisted = await service.get_current_state(session.id)
    assert raced_record is not None
    assert persisted is not None
    assert persisted.id == raced_record.id
    assert persisted.composer_meta is None or "guided_session" not in persisted.composer_meta


@pytest.mark.asyncio
async def test_guided_start_completed_retry_replays_after_later_freeform_head(tmp_path) -> None:
    from elspeth.web.sessions.protocol import CompositionStateData

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())
    payload = {"profile": "tutorial", "operation_id": operation_id}
    committed = client.post(f"/api/sessions/{session.id}/guided/start", json=payload)
    assert committed.status_code == 200
    later_freeform = await service.save_composition_state(
        session.id,
        CompositionStateData(
            sources={},
            nodes={},
            edges={},
            outputs={},
            metadata_={"name": "Later freeform", "description": ""},
            composer_meta=None,
        ),
        provenance="post_compose",
    )

    replay = client.post(f"/api/sessions/{session.id}/guided/start", json=payload)

    assert replay.status_code == 200
    assert replay.json() == committed.json()
    current = await service.get_current_state(session.id)
    assert current is not None
    assert current.id == later_freeform.id


@pytest.mark.asyncio
async def test_guided_start_empty_race_settles_exact_guided_winner(tmp_path) -> None:
    from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.routes._helpers import _initial_composition_state_with_guided_session

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    original_reserve = service.reserve_guided_operation
    winner_record = None

    async def reserve_after_guided_winner(**kwargs):
        nonlocal winner_record
        outcome = await original_reserve(**kwargs)
        if winner_record is None:
            winner = _materialize_first_turn(
                _initial_composition_state_with_guided_session(profile=TUTORIAL_PROFILE),
                app.state.catalog_service,
                app.state.payload_store,
            )
            assert winner.guided_session is not None
            winner_data = winner.to_dict()
            winner_record = await service.save_composition_state(
                session.id,
                CompositionStateData(
                    sources=winner_data["sources"],
                    nodes=winner_data["nodes"],
                    edges=winner_data["edges"],
                    outputs=winner_data["outputs"],
                    metadata_=winner_data["metadata"],
                    composer_meta={"guided_session": winner.guided_session.to_dict()},
                ),
                provenance="session_seed",
            )
        return outcome

    with patch.object(service, "reserve_guided_operation", side_effect=reserve_after_guided_winner):
        response = client.post(
            f"/api/sessions/{session.id}/guided/start",
            json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
        )

    assert response.status_code == 200
    assert winner_record is not None
    assert response.json()["composition_state"]["id"] == str(winner_record.id)
    current = await service.get_current_state(session.id)
    assert current is not None
    assert current.id == winner_record.id


@pytest.mark.asyncio
async def test_guided_start_late_empty_race_converges_inside_atomic_seed(tmp_path) -> None:
    from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.routes._helpers import _initial_composition_state_with_guided_session

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    original_seed = service.seed_or_complete_guided_start_operation
    winner_record = None

    async def seed_after_late_guided_winner(*args, **kwargs):
        nonlocal winner_record
        winner = _materialize_first_turn(
            _initial_composition_state_with_guided_session(profile=TUTORIAL_PROFILE),
            app.state.catalog_service,
            app.state.payload_store,
        )
        assert winner.guided_session is not None
        winner_data = winner.to_dict()
        winner_record = await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources=winner_data["sources"],
                nodes=winner_data["nodes"],
                edges=winner_data["edges"],
                outputs=winner_data["outputs"],
                metadata_=winner_data["metadata"],
                composer_meta={"guided_session": winner.guided_session.to_dict()},
            ),
            provenance="session_seed",
        )
        return await original_seed(*args, **kwargs)

    with patch.object(
        service,
        "seed_or_complete_guided_start_operation",
        side_effect=seed_after_late_guided_winner,
    ):
        response = client.post(
            f"/api/sessions/{session.id}/guided/start",
            json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
        )

    assert response.status_code == 200
    assert winner_record is not None
    assert response.json()["composition_state"]["id"] == str(winner_record.id)
    versions = await service.get_state_versions(session.id)
    assert [record.id for record in versions] == [winner_record.id]


@pytest.mark.asyncio
async def test_guided_start_replay_rejects_cross_session_result_locator(tmp_path) -> None:
    from elspeth.contracts.errors import AuditIntegrityError
    from elspeth.web.sessions.guided_operations import guided_operation_request_hash
    from elspeth.web.sessions.protocol import (
        CompositionStateData,
        GuidedCompositionStateResult,
        GuidedOperationCompleted,
    )
    from elspeth.web.sessions.schemas import StartGuidedRequest

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    foreign = await service.create_session("alice", "Other", "local")
    foreign_state = await service.save_composition_state(
        foreign.id,
        CompositionStateData(
            sources={},
            nodes={},
            edges={},
            outputs={},
            metadata_={"name": "Foreign", "description": ""},
            composer_meta=None,
        ),
        provenance="post_compose",
    )
    operation_id = str(uuid.uuid4())
    payload = {"profile": "tutorial", "operation_id": operation_id}
    first = client.post(f"/api/sessions/{session.id}/guided/start", json=payload)
    assert first.status_code == 200
    request_model = StartGuidedRequest.model_validate(payload)
    completed = await service.reserve_guided_operation(
        session_id=session.id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash=guided_operation_request_hash(
            session_id=session.id,
            kind="guided_start",
            request=request_model,
        ),
        actor="composer_route",
        lease_seconds=300,
    )
    assert isinstance(completed, GuidedOperationCompleted)
    corrupt_outcome = GuidedOperationCompleted(
        result=GuidedCompositionStateResult(state_id=foreign_state.id),
        response_hash=completed.response_hash,
    )

    with (
        patch.object(service, "get_guided_operation", return_value=corrupt_outcome),
        pytest.raises(AuditIntegrityError, match="Cross-session state reference rejected"),
    ):
        client.post(f"/api/sessions/{session.id}/guided/start", json=payload)


@pytest.mark.asyncio
async def test_guided_start_joins_active_operation_outside_compose_lock(tmp_path) -> None:
    from elspeth.web.sessions.guided_operations import guided_operation_request_hash
    from elspeth.web.sessions.protocol import GuidedOperationClaimed
    from elspeth.web.sessions.routes.guided_operations import guided_response_hash
    from elspeth.web.sessions.schemas import GetGuidedResponse, StartGuidedRequest

    app, service = _make_app(tmp_path, database_url=f"sqlite:///{tmp_path / 'active-join.db'}")
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    seeded = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
    )
    assert seeded.status_code == 200
    current = await service.get_current_state(session.id)
    assert current is not None
    operation_id = str(uuid.uuid4())
    payload = {"profile": "tutorial", "operation_id": operation_id}
    request_model = StartGuidedRequest.model_validate(payload)
    claim = await service.reserve_guided_operation(
        session_id=session.id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash=guided_operation_request_hash(
            session_id=session.id,
            kind="guided_start",
            request=request_model,
        ),
        actor="composer_route",
        lease_seconds=300,
    )
    assert isinstance(claim, GuidedOperationClaimed)

    joining = asyncio.create_task(
        asyncio.to_thread(
            client.post,
            f"/api/sessions/{session.id}/guided/start",
            json=payload,
        )
    )
    await asyncio.sleep(0.1)
    expected_response = GetGuidedResponse.model_validate_json(seeded.content)
    await service.complete_existing_state_guided_operation(
        claim.fence,
        state_id=current.id,
        expected_current_state_id=current.id,
        expected_current_state_version=current.version,
        actor="composer_route",
        response_hash_factory=lambda _record: guided_response_hash(expected_response),
    )
    joined = await asyncio.wait_for(joining, timeout=2)

    assert joined.status_code == 200
    assert joined.json() == seeded.json()


@pytest.mark.asyncio
async def test_guided_start_takes_over_expired_lease_and_stale_worker_cannot_settle(tmp_path) -> None:
    from sqlalchemy import text

    from elspeth.web.sessions.guided_operations import guided_operation_request_hash
    from elspeth.web.sessions.protocol import (
        GuidedOperationClaimed,
        GuidedOperationFenceLostError,
    )
    from elspeth.web.sessions.schemas import StartGuidedRequest

    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    seeded = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
    )
    assert seeded.status_code == 200
    operation_id = str(uuid.uuid4())
    payload = {"profile": "tutorial", "operation_id": operation_id}
    request_model = StartGuidedRequest.model_validate(payload)
    stale = await service.reserve_guided_operation(
        session_id=session.id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash=guided_operation_request_hash(
            session_id=session.id,
            kind="guided_start",
            request=request_model,
        ),
        actor="composer_route",
        lease_seconds=300,
    )
    assert isinstance(stale, GuidedOperationClaimed)
    with service._engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE guided_operations SET lease_expires_at = :expired WHERE session_id = :session_id AND operation_id = :operation_id"
            ),
            {
                "expired": datetime.now(UTC) - timedelta(seconds=1),
                "session_id": str(session.id),
                "operation_id": operation_id,
            },
        )

    with patch.object(service, "renew_guided_operation", wraps=service.renew_guided_operation) as renew:
        takeover = client.post(f"/api/sessions/{session.id}/guided/start", json=payload)

    assert takeover.status_code == 200
    assert takeover.json() == seeded.json()
    assert renew.await_count == 1
    with pytest.raises(GuidedOperationFenceLostError):
        await service.complete_existing_state_guided_operation(
            stale.fence,
            state_id=uuid.UUID(seeded.json()["composition_state"]["id"]),
            expected_current_state_id=uuid.UUID(seeded.json()["composition_state"]["id"]),
            expected_current_state_version=seeded.json()["composition_state"]["version"],
            actor="composer_route",
            response_hash_factory=lambda _record: "0" * 64,
        )


@pytest.mark.asyncio
async def test_guided_start_rejoins_after_fence_loss_without_polling_under_lock(tmp_path) -> None:
    from sqlalchemy import text

    from elspeth.web.sessions.protocol import GuidedOperationFenceLostError

    app, service = _make_app(tmp_path, database_url=f"sqlite:///{tmp_path / 'fence-loss.db'}")
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    original_renew = service.renew_guided_operation
    renew_calls = 0

    async def lose_first_fence(fence, *, actor, lease_seconds):
        nonlocal renew_calls
        renew_calls += 1
        if renew_calls == 1:
            with service._engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE guided_operations SET lease_expires_at = :expired "
                        "WHERE session_id = :session_id AND operation_id = :operation_id"
                    ),
                    {
                        "expired": datetime.now(UTC) - timedelta(seconds=1),
                        "session_id": str(fence.session_id),
                        "operation_id": fence.operation_id,
                    },
                )
            raise GuidedOperationFenceLostError(fence)
        return await original_renew(fence, actor=actor, lease_seconds=lease_seconds)

    with patch.object(service, "renew_guided_operation", side_effect=lose_first_fence):
        response = client.post(
            f"/api/sessions/{session.id}/guided/start",
            json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
        )

    assert response.status_code == 200
    assert renew_calls == 2
    with service._engine.connect() as conn:
        attempt = conn.execute(
            text("SELECT attempt FROM guided_operations WHERE session_id = :session_id AND kind = 'guided_start'"),
            {"session_id": str(session.id)},
        ).scalar_one()
    assert attempt == 2


@pytest.mark.asyncio
async def test_guided_start_repeated_request_cancellation_drains_terminal_failure(tmp_path) -> None:
    app, service = _make_app(tmp_path, database_url=f"sqlite:///{tmp_path / 'start-cancel.db'}")
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())
    renew_entered = asyncio.Event()
    failure_entered = asyncio.Event()
    release_failure = asyncio.Event()
    original_fail = service.fail_guided_operation

    async def blocked_renew(*_args, **_kwargs):
        renew_entered.set()
        await asyncio.Event().wait()

    async def blocked_failure(*args, **kwargs):
        failure_entered.set()
        await release_failure.wait()
        return await original_fail(*args, **kwargs)

    with (
        patch.object(service, "renew_guided_operation", side_effect=blocked_renew),
        patch.object(service, "fail_guided_operation", side_effect=blocked_failure),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            request_task = asyncio.create_task(
                client.post(
                    f"/api/sessions/{session.id}/guided/start",
                    json={"profile": "tutorial", "operation_id": operation_id},
                )
            )
            await asyncio.wait_for(renew_entered.wait(), timeout=5)
            request_task.cancel("operator cancelled guided start")
            await asyncio.wait_for(failure_entered.wait(), timeout=5)
            request_task.cancel("shutdown repeated guided start cancellation")
            await asyncio.sleep(0)
            assert not request_task.done()
            release_failure.set()
            with pytest.raises(asyncio.CancelledError, match="operator cancelled guided start") as caught:
                await asyncio.wait_for(request_task, timeout=5)

    assert caught.value.args == ("operator cancelled guided start",)
    assert await service.get_current_state(session.id) is None
    with service._engine.connect() as conn:
        operation = conn.execute(
            select(guided_operations_table.c.status, guided_operations_table.c.failure_code)
            .where(guided_operations_table.c.session_id == str(session.id))
            .where(guided_operations_table.c.operation_id == operation_id)
        ).one()
        failure_events = conn.execute(
            select(guided_operation_events_table.c.event_kind)
            .where(guided_operation_events_table.c.session_id == str(session.id))
            .where(guided_operation_events_table.c.operation_id == operation_id)
            .where(guided_operation_events_table.c.event_kind == "failed")
        ).all()
    assert operation.status == "failed"
    assert operation.failure_code == "request_cancelled"
    assert len(failure_events) == 1


@pytest.mark.asyncio
async def test_guided_start_cancellation_during_atomic_seed_drains_to_completed_replay(tmp_path) -> None:
    app, service = _make_app(tmp_path, database_url=f"sqlite:///{tmp_path / 'start-seed-cancel.db'}")
    session = await service.create_session("alice", "T", "local")
    operation_id = str(uuid.uuid4())
    payload = {"profile": "tutorial", "operation_id": operation_id}
    audit_inserted = threading.Event()
    release_worker = threading.Event()
    original_insert = service._insert_prepared_guided_audit_rows_on_connection

    def pause_after_audit_insert(*args, **kwargs):
        records = original_insert(*args, **kwargs)
        audit_inserted.set()
        if not release_worker.wait(timeout=5):
            raise TimeoutError("test did not release guided start settlement worker")
        return records

    with patch.object(service, "_insert_prepared_guided_audit_rows_on_connection", side_effect=pause_after_audit_insert):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            request_task = asyncio.create_task(client.post(f"/api/sessions/{session.id}/guided/start", json=payload))
            assert await asyncio.to_thread(audit_inserted.wait, 5), "guided start settlement worker did not start"
            request_task.cancel("operator cancelled atomic guided start")
            await asyncio.sleep(0)
            assert not request_task.done()
            release_worker.set()
            with pytest.raises(asyncio.CancelledError, match="operator cancelled atomic guided start") as caught:
                await asyncio.wait_for(request_task, timeout=5)
            replay = await asyncio.wait_for(
                client.post(f"/api/sessions/{session.id}/guided/start", json=payload),
                timeout=5,
            )

    assert caught.value.args == ("operator cancelled atomic guided start",)
    assert replay.status_code == 200, replay.json()
    with service._engine.connect() as conn:
        operation = conn.execute(
            select(
                guided_operations_table.c.status,
                guided_operations_table.c.failure_code,
                guided_operations_table.c.result_state_id,
            )
            .where(guided_operations_table.c.session_id == str(session.id))
            .where(guided_operations_table.c.operation_id == operation_id)
        ).one()
        state_count = conn.execute(
            select(composition_states_table.c.id).where(composition_states_table.c.session_id == str(session.id))
        ).all()
    assert operation.status == "completed"
    assert operation.failure_code is None
    assert replay.json()["composition_state"]["id"] == operation.result_state_id
    assert len(state_count) == 1


@pytest.mark.asyncio
async def test_guided_start_unowned_session_404(tmp_path) -> None:
    app, _service = _make_app(tmp_path, user_id="alice")
    client = TestClient(app)
    resp = client.post(
        f"/api/sessions/{uuid.uuid4()}/guided/start",
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_guided_respond_rejects_removed_stale_step_index_field(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
    )
    client.get(f"/api/sessions/{session.id}/guided")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/respond",
        json={"step_index": "step_3_transforms", "chosen": ["csv"]},
    )
    assert resp.status_code == 422
    assert any(error["loc"][-1] == "step_index" for error in resp.json()["detail"])


@pytest.mark.asyncio
async def test_guided_respond_rejects_removed_unknown_step_index_field(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
    )
    client.get(f"/api/sessions/{session.id}/guided")

    resp = client.post(
        f"/api/sessions/{session.id}/guided/respond",
        json={"step_index": "step_99_bogus", "chosen": ["csv"]},
    )
    assert resp.status_code == 422
    assert any(error["loc"][-1] == "step_index" for error in resp.json()["detail"])


@pytest.mark.asyncio
async def test_guided_respond_success_preserves_tutorial_profile(tmp_path) -> None:
    """A normal respond response still carries the persisted tutorial profile."""
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "T", "local")
    started = client.post(
        f"/api/sessions/{session.id}/guided/start",
        json={"profile": "tutorial", "operation_id": str(uuid.uuid4())},
    )
    turn = started.json()["next_turn"]

    resp = client.post(
        f"/api/sessions/{session.id}/guided/respond",
        json={
            "operation_id": str(uuid.uuid4()),
            "turn_token": turn["turn_token"],
            "chosen": ["inline_blob"],
        },
    )
    assert resp.status_code == 200
    profile = resp.json()["guided_session"]["profile"]
    assert profile is not None
    assert profile["advisor_checkpoints"] is False
    assert profile["bookends"] is True

    get_resp = client.get(f"/api/sessions/{session.id}/guided")
    assert get_resp.status_code == 200
    assert get_resp.json()["guided_session"]["profile"]["advisor_checkpoints"] is False
