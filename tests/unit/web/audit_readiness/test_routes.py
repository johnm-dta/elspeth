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
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

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
    app.include_router(create_audit_readiness_router())
    return TestClient(app)


def test_snapshot_does_not_flatten_unrelated_lookup_error_to_404() -> None:
    with (
        _client() as client,
        pytest.raises(LookupError, match="internal dict lookup exploded"),
    ):
        client.get(f"/api/sessions/{_SESSION_ID}/audit-readiness")
