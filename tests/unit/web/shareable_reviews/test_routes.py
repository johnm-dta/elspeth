"""Route tests for shareable-review endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretBytes

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.shareable_reviews.routes import create_shareable_reviews_router
from elspeth.web.shareable_reviews.service import CompositionNotRunnableError


@dataclass(frozen=True, slots=True)
class _SessionRecord:
    id: UUID
    user_id: str
    auth_provider_type: str


@dataclass(slots=True)
class _SessionService:
    session: _SessionRecord

    async def get_session(self, session_id: UUID) -> _SessionRecord:
        return self.session


@dataclass(slots=True)
class _ShareableReviewService:
    get_shareable_link_error: Exception

    async def get_shareable_link(self, *, session_id: UUID, user_id: str) -> object:
        raise self.get_shareable_link_error


def _settings() -> WebSettings:
    return WebSettings(
        auth_provider="local",
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=SecretBytes(b"\x00" * 32),
    )


async def _mock_user() -> UserIdentity:
    return UserIdentity(user_id="alice", username="alice")


def _app_with_share_service(shareable_review_service: object, *, session_id: UUID) -> FastAPI:
    app = FastAPI()
    app.state.settings = _settings()
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.shareable_review_service = shareable_review_service
    app.state.session_service = _SessionService(session=_SessionRecord(id=session_id, user_id="alice", auth_provider_type="local"))
    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_shareable_reviews_router())
    return app


@pytest.mark.asyncio
async def test_get_shareable_link_maps_unmarked_current_snapshot_to_409() -> None:
    session_id = uuid4()
    shareable_review_service = _ShareableReviewService(
        get_shareable_link_error=CompositionNotRunnableError(
            reason="not_marked_ready",
            detail="current composition state has not been marked ready for review",
        )
    )
    app = _app_with_share_service(shareable_review_service, session_id=session_id)

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get(f"/api/sessions/{session_id}/shareable-link")

    assert response.status_code == 409
    assert response.json()["detail"] == "current composition state has not been marked ready for review"
