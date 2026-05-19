"""Integration tests for tutorial backend routes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import structlog
from fastapi import FastAPI
from pydantic import SecretBytes
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_run_routes import create_tutorial_run_router
from elspeth.web.config import WebSettings
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.preferences.tutorial_cache import CANONICAL_SEED_PROMPT, TutorialCache, TutorialCacheEntry, tutorial_cache_key
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.integration.web.conftest import _make_session
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _settings(tmp_path: Path) -> WebSettings:
    (tmp_path / "runs").mkdir(exist_ok=True)
    return WebSettings(
        data_dir=tmp_path,
        landscape_url=f"sqlite:///{tmp_path}/runs/audit.db",
        composer_model="gpt-5.5",
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=SecretBytes(b"\x00" * 32),
    )


def _app(tmp_path: Path) -> FastAPI:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    settings = _settings(tmp_path)
    session_service = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.tutorial"),
    )

    app = FastAPI()
    app.state.settings = settings
    app.state.session_engine = engine
    app.state.session_service = session_service
    app.state.preferences_service = PreferencesService(engine)
    tutorial_cache_dir = settings.tutorial_cache_dir
    assert tutorial_cache_dir is not None
    app.state.tutorial_cache = TutorialCache(cache_dir=tutorial_cache_dir)

    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_preferences_router())
    app.include_router(create_tutorial_run_router())
    app.include_router(create_session_router())
    return app


def _seed_session_with_state(app: FastAPI) -> UUID:
    session_id = uuid4()
    with app.state.session_engine.begin() as conn:
        _make_session(conn, session_id=str(session_id), user_id="alice")
    asyncio.run(
        app.state.session_service.save_composition_state(
            session_id,
            CompositionStateData(),
            provenance="session_seed",
        )
    )
    return session_id


def _seed_canonical_cache(app: FastAPI) -> str:
    model_id = app.state.settings.composer_model
    key = tutorial_cache_key(CANONICAL_SEED_PROMPT, model_id)
    app.state.tutorial_cache.store(
        TutorialCacheEntry(
            canonical_prompt=CANONICAL_SEED_PROMPT,
            model_id=model_id,
            cached_at=datetime(2026, 5, 15, tzinfo=UTC),
            rows=[{"url": "ato.gov.au", "score": 5, "rationale": "clear"}],
            source_data_hash="a7f3e2cached",
            llm_call_count=5,
            pipeline_yaml=("source:\n  plugin: 'null'\ntransforms:\n  keep:\n    plugin: passthrough\nsinks:\n  out:\n    plugin: json\n"),
        )
    )
    return key


def test_post_run_cache_hit_creates_current_session_run_and_audit_story(tmp_path: Path) -> None:
    app = _app(tmp_path)
    cache_key = _seed_canonical_cache(app)
    session_id = _seed_session_with_state(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": str(session_id), "prompt": CANONICAL_SEED_PROMPT},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["seeded_from_cache"] is True
    assert body["cache_key"] == cache_key
    assert body["output"]["source_data_hash"] == "a7f3e2cached"

    story = client.get(f"/api/sessions/{session_id}/runs/{body['run_id']}/audit-story")
    assert story.status_code == 200
    story_body = story.json()
    assert story_body["run_id"] == body["run_id"]
    assert story_body["session_id"] == str(session_id)
    assert story_body["llm_call_count"] == 0
    assert story_body["seeded_from_cache"] is True
    assert story_body["cache_key"] == cache_key


def test_post_run_unknown_session_returns_404(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _seed_canonical_cache(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": str(uuid4()), "prompt": CANONICAL_SEED_PROMPT},
    )

    assert response.status_code == 404


def test_post_run_rejects_extra_fields(tmp_path: Path) -> None:
    app = _app(tmp_path)
    session_id = _seed_session_with_state(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": str(session_id), "prompt": CANONICAL_SEED_PROMPT, "rogue": True},
    )

    assert response.status_code == 422


def test_delete_orphans_soft_renames_incomplete_tutorial_sessions(tmp_path: Path) -> None:
    app = _app(tmp_path)
    orphan_id = uuid4()
    keep_id = uuid4()
    with app.state.session_engine.begin() as conn:
        _make_session(conn, session_id=str(orphan_id), user_id="alice", title="hello-world (cool government pages)")
        _make_session(conn, session_id=str(keep_id), user_id="alice", title="ordinary session")
    client = TestClient(app)

    response = client.delete("/api/tutorial/orphans")

    assert response.status_code == 200
    assert response.json() == {"deleted_count": 1}
    renamed = asyncio.run(app.state.session_service.get_session(orphan_id))
    kept = asyncio.run(app.state.session_service.get_session(keep_id))
    assert renamed.title.startswith("abandoned-hello-world (cool government pages)-")
    assert kept.title == "ordinary session"
