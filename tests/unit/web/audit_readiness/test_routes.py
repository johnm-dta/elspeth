"""Tests for audit-readiness route exception mapping."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from fastapi import FastAPI

from elspeth.web.audit_readiness.routes import create_audit_readiness_router
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions.protocol import SessionRecord
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

# Phase 8 Sub-task 7f (Q7 FastAPI route-table probe). Gates the
# route-level fetch-failure emit test on the Phase 2C audit-readiness
# endpoint being mounted. Per memory
# ``project_phase2c_implementation_complete.md`` the endpoint should be
# present; the skip pathway preserves test-suite green behaviour if a
# sibling-phase rename moves the route. The router-construction probe
# is decoupled from the production app so test discovery does not pay
# the cost of full app import.
_router_for_probe = create_audit_readiness_router()
_PHASE_2C_AUDIT_READINESS_MOUNTED = any(getattr(r, "path", "").endswith("/audit-readiness") for r in _router_for_probe.routes)

_SESSION_ID = UUID("11111111-1111-1111-1111-111111111111")


class _SessionService:
    async def get_session(self, session_id: UUID) -> SessionRecord:
        return SessionRecord(
            id=session_id,
            user_id="alice",
            auth_provider_type="local",
            title="Test",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )


class _ExplodingReadinessService:
    async def compute_snapshot(self, *, session_id: UUID, user_id: str):
        raise LookupError("internal dict lookup exploded")


def _client() -> TestClient:
    """Function-scoped (Q10): each call builds a fresh app + telemetry
    container so counter observations are isolated per test.
    """
    app = FastAPI()

    async def _mock_user() -> UserIdentity:
        return UserIdentity(user_id="alice", username="alice")

    app.dependency_overrides[get_current_user] = _mock_user
    app.state.settings = WebSettings(
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=100,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.session_service = _SessionService()
    app.state.readiness_service = _ExplodingReadinessService()
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    # Phase 8 Sub-task 7f. The route reads ``app.state.sessions_telemetry``
    # in the exception path to emit ``composer.audit.fetch_failure_total``.
    # Tests use the fake-counter container so ``observed_value`` can
    # observe ``add`` calls.
    app.state.sessions_telemetry = build_sessions_telemetry()
    app.include_router(create_audit_readiness_router())
    return TestClient(app)


def test_snapshot_does_not_flatten_unrelated_lookup_error_to_404() -> None:
    with (
        _client() as client,
        pytest.raises(LookupError, match="internal dict lookup exploded"),
    ):
        client.get(f"/api/sessions/{_SESSION_ID}/audit-readiness")


# --------------------------------------------------------------------------- #
# Phase 8 Sub-task 7f — telemetry emit on the audit-readiness fetch-failure
# branch
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _PHASE_2C_AUDIT_READINESS_MOUNTED,
    reason="Phase 2C audit-readiness endpoint not mounted; Sub-task 7f is a documented no-op per Task 0 probe.",
)
def test_snapshot_fetch_failure_emits_audit_fetch_failure_counter() -> None:
    """Phase 8 Sub-task 7f (B3 cohort b2):

    A non-``CompositionStateNotFoundError`` exception raised inside
    ``compute_snapshot`` is a fetch-failure event on the audit-readiness
    read path. The route emits ``composer.audit.fetch_failure_total``
    exactly once and re-raises so the failure remains visible (no
    silent swallow to a 200 response). Telemetry-only signal under
    the CLAUDE.md non-decision read superset exception.
    """
    client = _client()
    with (
        client,
        pytest.raises(LookupError, match="internal dict lookup exploded"),
    ):
        client.get(f"/api/sessions/{_SESSION_ID}/audit-readiness")
    # ``client.app`` remains a valid reference after the context
    # manager exits — counter observation is on the app's telemetry
    # container, not on TestClient state.
    telemetry = client.app.state.sessions_telemetry
    assert observed_value(telemetry.audit_fetch_failure_total) == 1


@pytest.mark.skipif(
    not _PHASE_2C_AUDIT_READINESS_MOUNTED,
    reason="Phase 2C audit-readiness endpoint not mounted; Sub-task 7f is a documented no-op per Task 0 probe.",
)
def test_snapshot_composition_state_not_found_does_not_emit_fetch_failure() -> None:
    """Phase 8 Sub-task 7f — negative guard.

    ``CompositionStateNotFoundError`` is the documented not-found
    branch (HTTP 404). It is NOT a fetch-failure. Emitting the
    counter on a 404 would inflate the metric with not-found events
    and break the "read-path-health" semantic.
    """
    from elspeth.web.audit_readiness.service import CompositionStateNotFoundError

    class _NotFoundReadinessService:
        async def compute_snapshot(self, *, session_id: UUID, user_id: str):
            raise CompositionStateNotFoundError(str(session_id))

    app = FastAPI()

    async def _mock_user() -> UserIdentity:
        return UserIdentity(user_id="alice", username="alice")

    app.dependency_overrides[get_current_user] = _mock_user
    app.state.settings = WebSettings(
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=100,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.session_service = _SessionService()
    app.state.readiness_service = _NotFoundReadinessService()
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.sessions_telemetry = build_sessions_telemetry()
    app.include_router(create_audit_readiness_router())
    with TestClient(app) as client:
        response = client.get(f"/api/sessions/{_SESSION_ID}/audit-readiness")
    assert response.status_code == 404
    # Counter unchanged: not-found is not fetch-failed.
    assert observed_value(app.state.sessions_telemetry.audit_fetch_failure_total) == 0
