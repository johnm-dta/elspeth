"""Tutorial run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_models import TutorialOrphanCleanupResponse, TutorialRunRequest, TutorialRunResponse
from elspeth.web.composer.tutorial_service import cleanup_tutorial_orphans, run_tutorial_pipeline


def create_tutorial_run_router() -> APIRouter:
    """Create the tutorial API router."""
    router = APIRouter(prefix="/api/tutorial", tags=["tutorial"])

    @router.post("/run", response_model=TutorialRunResponse)
    async def run_tutorial(
        body: TutorialRunRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> TutorialRunResponse:
        return await run_tutorial_pipeline(
            request=request,
            user=user,
            session_id=str(body.session_id),
            prompt=body.prompt,
        )

    @router.delete("/orphans", response_model=TutorialOrphanCleanupResponse)
    async def delete_tutorial_orphans(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> TutorialOrphanCleanupResponse:
        return await cleanup_tutorial_orphans(request=request, user=user)

    return router
