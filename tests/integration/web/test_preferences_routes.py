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
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema


def _make_app(user_id: str | None) -> FastAPI:
    """Build a minimal FastAPI app wired for preferences tests."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    app = FastAPI()
    app.state.preferences_service = PreferencesService(engine)
    app.state.session_engine = engine

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
    client_as_alice: TestClient,
    client_as_bob: TestClient,
) -> None:
    """Route-level cross-user isolation: alice's prefs are invisible to bob.

    A service-layer test (``test_users_are_isolated``) also covers this,
    but a route bug could leak across users while service tests stay green.
    """
    # Alice sets freeform.
    resp = client_as_alice.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert resp.status_code == 200

    # Bob — on a separate engine/app — still sees the guided default.
    bob_resp = client_as_bob.get("/api/composer-preferences")
    assert bob_resp.status_code == 200
    assert bob_resp.json()["default_mode"] == "guided"


def test_db_unavailable_returns_500() -> None:
    """Infrastructure failure → 500 (not a Tier-1 corruption event)."""
    from unittest.mock import AsyncMock, MagicMock

    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    broken_service: MagicMock = MagicMock()
    broken_service.get_composer_preferences = AsyncMock(side_effect=Exception("DB connection refused"))
    app.state.preferences_service = broken_service
    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_preferences_router())

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/composer-preferences")
    assert response.status_code == 500
