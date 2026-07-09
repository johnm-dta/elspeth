"""Integration tests for /api/composer-preferences.

Fixtures are self-contained (no shared conftest fixtures); modelled on
``tests/integration/web/conftest.py``'s ``composer_test_client``.

Three fixtures:
  - ``client_as_alice`` — authenticated as ``user_id="alice"``.
  - ``client_as_bob`` — authenticated as ``user_id="bob"`` (required for
    cross-user isolation test).
  - ``client_anonymous`` — auth override raises ``HTTPException(401)``
    directly. The real ``get_current_user`` (``auth/middleware.py:38-39``)
    reads ``app.state.auth_audit_recorder`` and ``app.state.settings``
    *before* checking the Authorization header, so leaving it in place
    against a bare ``FastAPI()`` would raise ``AttributeError`` → 500.
    The override asserts route-layer auth-required behaviour without
    standing up the full auth surface.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI, HTTPException
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _make_app(
    user_id: str | None,
    *,
    engine: object | None = None,
    rate_limit: int = 100,
) -> FastAPI:
    """Build a minimal FastAPI app wired for preferences tests.

    ``engine``: pass a pre-built engine to share storage across multiple
    client apps (Panel C3 — cross-user isolation must observe both users
    on the same DB).

    ``rate_limit``: per-user PATCH rate. Default of 100/min is high enough
    that ordinary tests don't bump into it; the rate-limit test overrides
    to a low value.
    """
    if engine is None:
        engine = create_session_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        initialize_session_schema(engine)
    app = FastAPI()
    app.state.preferences_service = PreferencesService(engine)
    app.state.session_engine = engine
    app.state.rate_limiter = ComposerRateLimiter(limit=rate_limit)
    # Phase 8 Task 2: the PATCH route emits ``record_mode_opted_*``
    # via ``request.app.state.sessions_telemetry``. Attach a fresh
    # fake-counter container per app build (function-scoped per Q10).
    app.state.sessions_telemetry = build_sessions_telemetry()

    if user_id is not None:
        identity = UserIdentity(user_id=user_id, username=user_id)

        async def _mock_user() -> UserIdentity:
            return identity

        app.dependency_overrides[get_current_user] = _mock_user
    else:

        async def _unauthenticated() -> UserIdentity:
            raise HTTPException(status_code=401, detail="Not authenticated")

        app.dependency_overrides[get_current_user] = _unauthenticated

    app.include_router(create_preferences_router())
    return app


@pytest.fixture
def client_as_alice() -> Iterator[TestClient]:
    yield TestClient(_make_app("alice"))


@pytest.fixture
def client_as_bob() -> Iterator[TestClient]:
    yield TestClient(_make_app("bob"))


@pytest.fixture
def shared_engine() -> Iterator[object]:
    """Panel C3: a single engine shared across multiple TestClients so a
    same-DB SQL-scoping bug (e.g. forgotten WHERE user_id=...) would actually
    surface in the cross-user isolation test. The earlier per-client engines
    made the test tautological — alice and bob were on different SQLite
    databases, so isolation 'passed' for the wrong reason."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def cross_user_alice(shared_engine: object) -> Iterator[TestClient]:
    yield TestClient(_make_app("alice", engine=shared_engine))


@pytest.fixture
def cross_user_bob(shared_engine: object) -> Iterator[TestClient]:
    yield TestClient(_make_app("bob", engine=shared_engine))


@pytest.fixture
def client_anonymous() -> Iterator[TestClient]:
    # raise_server_exceptions left at the default (True): the 401 override
    # raises HTTPException, which FastAPI converts to a 401 response — no
    # server exception should propagate. If one does, the test fails loudly
    # rather than silently 500.
    yield TestClient(_make_app(None))


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


def test_get_returns_guided_default_for_brand_new_user(client_as_alice: TestClient) -> None:
    response = client_as_alice.get("/api/composer-preferences")
    assert response.status_code == 200
    body = response.json()
    assert body["default_mode"] == "guided"
    assert body["banner_dismissed_at"] is None
    assert body["tutorial_completed_at"] is None


def test_patch_updates_default_mode(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert response.status_code == 200
    assert response.json()["default_mode"] == "freeform"

    follow_up = client_as_alice.get("/api/composer-preferences")
    assert follow_up.json()["default_mode"] == "freeform"


def test_patch_persists_banner_dismissal(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"banner_dismissed_at": "2026-05-15T12:00:00Z"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["banner_dismissed_at"] is not None
    # SQLite strips tzinfo; the value is preserved either way.
    assert "2026-05-15T12:00:00" in body["banner_dismissed_at"]


def test_patch_sets_tutorial_completed_at(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "2026-05-15T12:30:00Z"},
    )
    assert response.status_code == 200
    assert "2026-05-15T12:30:00" in response.json()["tutorial_completed_at"]


def test_patch_atomic_finalisation_payload(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={
            "default_mode": "freeform",
            "tutorial_completed_at": "2026-05-15T12:35:00Z",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["default_mode"] == "freeform"
    assert "2026-05-15T12:35:00" in body["tutorial_completed_at"]


def test_patch_with_explicit_null_clears_banner_dismissal(client_as_alice: TestClient) -> None:
    """PATCH ``{"banner_dismissed_at": null}`` re-shows the banner. Symmetric
    with the tutorial-clear route test below."""
    set_response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"banner_dismissed_at": "2026-05-15T12:45:00Z"},
    )
    assert set_response.status_code == 200
    assert set_response.json()["banner_dismissed_at"] is not None

    clear_response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"banner_dismissed_at": None},
    )
    assert clear_response.status_code == 200
    assert clear_response.json()["banner_dismissed_at"] is None

    follow_up = client_as_alice.get("/api/composer-preferences")
    assert follow_up.json()["banner_dismissed_at"] is None


def test_patch_with_explicit_null_clears_tutorial(client_as_alice: TestClient) -> None:
    set_response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "2026-05-15T12:40:00Z"},
    )
    assert set_response.status_code == 200
    assert set_response.json()["tutorial_completed_at"] is not None

    clear_response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": None},
    )
    assert clear_response.status_code == 200
    assert clear_response.json()["tutorial_completed_at"] is None


def test_patch_rejects_non_datetime_tutorial(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "yesterday"},
    )
    assert response.status_code == 422


def test_patch_rejects_invalid_mode(client_as_alice: TestClient) -> None:
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"default_mode": "kiosk"},
    )
    assert response.status_code == 422


def test_patch_rejects_unknown_field(client_as_alice: TestClient) -> None:
    """Extra fields must 422: a typo in the field name must not silently no-op."""
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"default_modd": "freeform"},  # typo
    )
    assert response.status_code == 422


def test_get_requires_auth(client_anonymous: TestClient) -> None:
    response = client_anonymous.get("/api/composer-preferences")
    assert response.status_code == 401


def test_patch_requires_auth(client_anonymous: TestClient) -> None:
    response = client_anonymous.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert response.status_code == 401


def test_users_cannot_see_each_others_preferences(
    cross_user_alice: TestClient,
    cross_user_bob: TestClient,
) -> None:
    """Route-level cross-user isolation: alice's PATCH must NOT be visible
    to bob's GET when both clients share the same database engine.

    Panel C3: previously the alice and bob clients held *independent*
    in-memory SQLite engines, so isolation 'passed' for the wrong reason
    — bob's GET couldn't see alice's PATCH because bob's engine had no
    rows for anyone. A real same-DB SQL-scoping bug (e.g. a forgotten
    ``WHERE user_id=...``) would not have surfaced. The shared_engine
    fixture below puts both clients on one engine so the test actually
    exercises route-layer scoping.
    """
    # Alice sets freeform on the shared DB.
    resp = cross_user_alice.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert resp.status_code == 200

    # Bob on the SAME DB still sees the guided default.
    bob_resp = cross_user_bob.get("/api/composer-preferences")
    assert bob_resp.status_code == 200
    assert bob_resp.json()["default_mode"] == "guided"


def test_patch_enforces_rate_limit_returns_429() -> None:
    """Panel C1: PATCH must call the shared ComposerRateLimiter."""
    app = _make_app("alice-rate-limit", rate_limit=2)
    client = TestClient(app)

    # Two PATCHes inside the 60s window succeed.
    for _ in range(2):
        ok = client.patch("/api/composer-preferences", json={"default_mode": "freeform"})
        assert ok.status_code == 200, ok.text

    # Third within the window must 429.
    over = client.patch("/api/composer-preferences", json={"default_mode": "guided"})
    assert over.status_code == 429, over.text


def test_db_unavailable_returns_503() -> None:
    """Infrastructure failure → 503 via the app's OperationalError handler.

    Panel U3: the earlier test raised a bare ``Exception`` and asserted
    500 — a contract the production app does NOT promise. ``app.py:628``
    installs an ``OperationalError`` exception handler that returns 503
    with ``error_type=database_unavailable``; that handler is what
    a real DB outage exercises. The test now mirrors that contract.

    A bare ``Exception`` would bypass the handler and surface as a
    generic 500; that's a different code path (programmer error / bug),
    not the DB-down user-visible outcome this test is meant to assert.
    """
    from sqlalchemy.exc import OperationalError

    class _UnavailablePreferencesService:
        async def get_composer_preferences(self, _user_id: str) -> object:
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.state.preferences_service = _UnavailablePreferencesService()
    app.dependency_overrides[get_current_user] = _mock_user

    # Register the production OperationalError handler — the test app
    # would otherwise let the exception escape as a generic 500.
    from sqlalchemy.exc import OperationalError as _OpErr

    @app.exception_handler(_OpErr)
    async def _db_unavailable_handler(_req: object, _exc: _OpErr) -> object:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={
                "detail": "Database is currently unavailable. Please retry in a moment.",
                "error_type": "database_unavailable",
            },
        )

    app.include_router(create_preferences_router())

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/composer-preferences")
    assert response.status_code == 503
    assert response.json()["error_type"] == "database_unavailable"


# ── Tutorial resume-state persistence (elspeth-918f4434b3) ─────────────────


def test_get_default_has_no_tutorial_progress(client_as_alice: TestClient) -> None:
    body = client_as_alice.get("/api/composer-preferences").json()
    assert body["tutorial_stage"] is None
    assert body["tutorial_session_id"] is None
    assert body["tutorial_run_id"] is None
    assert body["tutorial_source_data_hash"] is None


def test_patch_persists_tutorial_progress_round_trip(client_as_alice: TestClient) -> None:
    """The frontend PATCHes stage + session on every stage transition; a
    reload GETs them back and resumes at the persisted stage."""
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_stage": "guided", "tutorial_session_id": "sess-http-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tutorial_stage"] == "guided"
    assert body["tutorial_session_id"] == "sess-http-1"

    follow_up = client_as_alice.get("/api/composer-preferences").json()
    assert follow_up["tutorial_stage"] == "guided"
    assert follow_up["tutorial_session_id"] == "sess-http-1"


def test_patch_rejects_invalid_tutorial_stage(client_as_alice: TestClient) -> None:
    """'welcome' is deliberately outside the persisted vocabulary — nothing
    has started; NULL is the no-in-progress state."""
    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_stage": "welcome", "tutorial_session_id": "sess-http-2"},
    )
    assert response.status_code == 422


def test_e2e_reset_recipe_clears_tutorial_progress(client_as_alice: TestClient) -> None:
    """The e2e harness reset recipe — PATCH {"tutorial_completed_at": null,
    "default_mode": "guided"} — must restart the tutorial cleanly at Welcome
    even when a resume stage lingers from an interrupted tutorial."""
    client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_stage": "run", "tutorial_session_id": "sess-http-3"},
    )

    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": None, "default_mode": "guided"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tutorial_completed_at"] is None
    assert body["tutorial_stage"] is None
    assert body["tutorial_session_id"] is None

    follow_up = client_as_alice.get("/api/composer-preferences").json()
    assert follow_up["tutorial_stage"] is None
    assert follow_up["tutorial_session_id"] is None


def test_completion_clears_tutorial_progress_over_http(client_as_alice: TestClient) -> None:
    """Skip's immediate opt-out persist (tutorial_completed_at set) also
    terminates the in-progress resume state."""
    client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_stage": "graduation", "tutorial_session_id": "sess-http-4"},
    )

    response = client_as_alice.patch(
        "/api/composer-preferences",
        json={"tutorial_completed_at": "2026-05-15T13:00:00Z"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tutorial_completed_at"] is not None
    assert body["tutorial_stage"] is None
    assert body["tutorial_session_id"] is None
