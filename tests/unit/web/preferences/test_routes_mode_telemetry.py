"""Route-level telemetry tests for Phase 8 Task 2 Sites 1+2.

PATCH /api/composer-preferences must emit:
  - ``record_mode_opted_out`` when ``default_mode == "freeform"`` in the
    response (i.e. the PATCH body included ``default_mode`` and the
    post-state is freeform).
  - ``record_mode_opted_in`` when ``default_mode == "guided"`` in the
    response.
  - **Neither** when the PATCH did not include ``default_mode``
    (banner-dismissal-only PATCH).

Semantic pinning (B3-r3, load-bearing). The emit is a **set-rate**, not
a transition-rate: a PATCH with ``{"default_mode": "freeform"}`` against
a session whose prior was already ``freeform`` still fires the
``record_mode_opted_out`` emit, because ``mode_changed=True`` means
"PATCH body included ``default_mode``", NOT "post-state differs from
prior". The ``test_patch_same_default_mode_value_still_emits`` test
pins that contract — if a future PR silently switches the route to
``transition.prior.default_mode != transition.current.default_mode``,
this test fails.

Fixture discipline (Q10): function-scoped ``sessions_telemetry``
container per test — every call to ``_make_app()`` constructs a fresh
``build_sessions_telemetry()``. The ``_FakeCounter.calls`` list does not
accumulate across tests.
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
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _make_app(user_id: str | None = "alice") -> FastAPI:
    """Build a minimal FastAPI app wired for preferences-route tests.

    Fresh in-memory SQLite engine + fresh ``sessions_telemetry``
    container per call (Q10 — function-scoped, no cross-test
    accumulation).
    """
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    app = FastAPI()
    app.state.preferences_service = PreferencesService(engine)
    app.state.session_engine = engine
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
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
def client() -> Iterator[TestClient]:
    yield TestClient(_make_app("alice"))


# ---------------------------------------------------------------------------
# Site 1 — record_mode_opted_out
# ---------------------------------------------------------------------------


def test_patch_with_default_mode_freeform_emits_mode_opted_out_total(
    client: TestClient,
) -> None:
    """PATCH ``{"default_mode": "freeform"}`` increments
    ``mode_opted_out_total`` and leaves ``mode_opted_in_total`` at 0."""
    telemetry = client.app.state.sessions_telemetry
    assert observed_value(telemetry.mode_opted_out_total) == 0
    assert observed_value(telemetry.mode_opted_in_total) == 0

    response = client.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert response.status_code == 200

    assert observed_value(telemetry.mode_opted_out_total) == 1
    assert observed_value(telemetry.mode_opted_in_total) == 0


# ---------------------------------------------------------------------------
# Site 2 — record_mode_opted_in
# ---------------------------------------------------------------------------


def test_patch_with_default_mode_guided_emits_mode_opted_in_total(
    client: TestClient,
) -> None:
    """PATCH ``{"default_mode": "guided"}`` increments
    ``mode_opted_in_total`` and leaves ``mode_opted_out_total`` at 0."""
    telemetry = client.app.state.sessions_telemetry

    response = client.patch(
        "/api/composer-preferences",
        json={"default_mode": "guided"},
    )
    assert response.status_code == 200

    assert observed_value(telemetry.mode_opted_in_total) == 1
    assert observed_value(telemetry.mode_opted_out_total) == 0


# ---------------------------------------------------------------------------
# Negative — banner-dismissal-only PATCH must NOT emit either counter
# ---------------------------------------------------------------------------


def test_patch_without_default_mode_emits_neither_mode_counter(
    client: TestClient,
) -> None:
    """A PATCH that only sets ``banner_dismissed_at`` must not fire
    either mode-related counter (the ``mode_changed`` field-presence
    flag is False in this case)."""
    telemetry = client.app.state.sessions_telemetry

    response = client.patch(
        "/api/composer-preferences",
        json={"banner_dismissed_at": "2026-05-19T12:00:00Z"},
    )
    assert response.status_code == 200

    assert observed_value(telemetry.mode_opted_out_total) == 0
    assert observed_value(telemetry.mode_opted_in_total) == 0


# ---------------------------------------------------------------------------
# Set-rate semantic (B3-r3 — load-bearing)
# ---------------------------------------------------------------------------


def test_patch_same_default_mode_value_still_emits(client: TestClient) -> None:
    """Setting ``default_mode=freeform`` on a user whose prior was
    already ``freeform`` MUST still fire ``record_mode_opted_out``.

    The Phase 8 plan B3-r3 caveat is explicit: ``mode_changed=True``
    means "the PATCH body included the ``default_mode`` field", NOT
    "the value differs from the prior". This counter is a set-rate,
    not a transition-rate. Inferring "changed" from
    ``transition.prior.default_mode != transition.current.default_mode``
    would silently convert it to a transition-rate and break
    §"Account-level scope narrowing (B2.b — load-bearing)" — this
    test would fail in that regression.
    """
    telemetry = client.app.state.sessions_telemetry

    # First PATCH establishes the prior=freeform state.
    resp1 = client.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert resp1.status_code == 200
    assert observed_value(telemetry.mode_opted_out_total) == 1

    # Second PATCH re-asserts the same value. Even though the stored
    # row's ``default_mode`` did not change, the emit MUST fire again
    # — this is the set-rate semantic.
    resp2 = client.patch(
        "/api/composer-preferences",
        json={"default_mode": "freeform"},
    )
    assert resp2.status_code == 200
    assert observed_value(telemetry.mode_opted_out_total) == 2
    assert observed_value(telemetry.mode_opted_in_total) == 0
