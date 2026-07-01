"""Integration tests for tutorial backend routes.

The tutorial run is always a LIVE execution through the normal
``ExecutionService`` (no cached/replayed fast path). These route-level tests
mock ``_run_live_tutorial`` so they exercise the route's own responsibilities —
session ownership, rate limiting, request validation, and response shape —
without standing up the full execution backbone. The live projection itself is
covered by ``tests/unit/web/composer/test_tutorial_service.py`` and the audit
story by ``tests/unit/web/sessions/test_audit_story_service.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import structlog
from fastapi import FastAPI
from pydantic import SecretBytes
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer import tutorial_service as tutorial_service_module
from elspeth.web.composer.tutorial_models import TutorialRunOutput, TutorialRunResponse
from elspeth.web.composer.tutorial_run_routes import create_tutorial_run_router
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import PreferencesService
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
    app.state.rate_limiter = ComposerRateLimiter(limit=settings.composer_rate_limit_per_minute)

    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_preferences_router())
    app.include_router(create_tutorial_run_router())
    app.include_router(create_session_router())
    return app


def _seed_session_with_state(app: FastAPI) -> UUID:
    """Seed a session with a minimal composition state owned by ``alice``."""
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


def _install_fake_live_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str = "11111111-1111-1111-1111-111111111111",
) -> dict[str, int]:
    """Replace ``_run_live_tutorial`` with a stub that records its call count.

    The live execution backbone is out of scope for these route tests; the stub
    returns a fixed response with ``seeded_from_cache=False`` (the only shape the
    cache-free tutorial run produces) so the route's wiring can be asserted.
    """
    calls = {"count": 0}
    response = TutorialRunResponse(
        run_id=run_id,
        output=TutorialRunOutput(
            rows=({"url": "https://example.gov.au", "summary": "A page."},),
            source_data_hash="a" * 64,
        ),
        seeded_from_cache=False,
        cache_key=None,
    )

    class _FakeLiveRun:
        def __init__(self) -> None:
            self.response = response

    async def _fake_run_live_tutorial(**_kwargs: Any) -> _FakeLiveRun:
        calls["count"] += 1
        return _FakeLiveRun()

    monkeypatch.setattr(tutorial_service_module, "_run_live_tutorial", _fake_run_live_tutorial)
    return calls


def test_post_run_executes_live_and_returns_not_seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path)
    session_id = _seed_session_with_state(app)
    calls = _install_fake_live_run(monkeypatch)
    client = TestClient(app)

    response = client.post("/api/tutorial/run", json={"session_id": str(session_id)})

    assert response.status_code == 200
    body = response.json()
    # The cache-free tutorial run is always a real execution: never seeded.
    assert body["seeded_from_cache"] is False
    assert body["cache_key"] is None
    assert body["output"]["rows"] == [{"url": "https://example.gov.au", "summary": "A page."}]
    assert calls["count"] == 1


def test_post_run_rate_limiter_blocks_before_live_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path)
    app.state.rate_limiter = ComposerRateLimiter(limit=1)
    first_session_id = _seed_session_with_state(app)
    second_session_id = _seed_session_with_state(app)
    calls = _install_fake_live_run(monkeypatch)
    client = TestClient(app)

    allowed = client.post("/api/tutorial/run", json={"session_id": str(first_session_id)})
    assert allowed.status_code == 200

    throttled = client.post("/api/tutorial/run", json={"session_id": str(second_session_id)})
    assert throttled.status_code == 429
    assert throttled.json()["detail"]["error_type"] == "rate_limited"
    # The second request is throttled in the route BEFORE the service runs the
    # live pipeline, so only the first request reached _run_live_tutorial.
    assert calls["count"] == 1


def test_post_run_unknown_session_returns_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path)
    calls = _install_fake_live_run(monkeypatch)
    client = TestClient(app)

    response = client.post("/api/tutorial/run", json={"session_id": str(uuid4())})

    assert response.status_code == 404
    # Ownership is verified before the live run, so an unknown session never
    # reaches execution.
    assert calls["count"] == 0


def test_post_run_rejects_extra_fields(tmp_path: Path) -> None:
    app = _app(tmp_path)
    session_id = _seed_session_with_state(app)
    client = TestClient(app)

    response = client.post(
        "/api/tutorial/run",
        json={"session_id": str(session_id), "rogue": True},
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
