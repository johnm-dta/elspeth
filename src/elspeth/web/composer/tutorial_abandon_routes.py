"""Best-effort tutorial abandon beacon endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_telemetry import record_tutorial_abandoned


def create_tutorial_abandon_router() -> APIRouter:
    """Create the tutorial abandon-beacon router."""
    router = APIRouter(prefix="/api/tutorial", tags=["tutorial"])

    @router.post("/abandon", status_code=204)
    async def abandon_tutorial(
        _user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> Response:
        record_tutorial_abandoned()
        return Response(status_code=204)

    return router
