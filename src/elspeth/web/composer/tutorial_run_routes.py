"""Tutorial run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer.tutorial_models import (
    TutorialCancelRequest,
    TutorialCancelResponse,
    TutorialOrphanCleanupResponse,
    TutorialRunRequest,
    TutorialRunResponse,
)
from elspeth.web.composer.tutorial_service import cancel_tutorial_run, cleanup_tutorial_orphans, run_tutorial_pipeline
from elspeth.web.middleware.rate_limit import ComposerRateLimiter, get_rate_limiter


def create_tutorial_run_router() -> APIRouter:
    """Create the tutorial API router."""
    router = APIRouter(prefix="/api/tutorial", tags=["tutorial"])

    @router.post("/run", response_model=TutorialRunResponse)
    async def run_tutorial(
        body: TutorialRunRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    ) -> TutorialRunResponse:
        await rate_limiter.check(user.user_id)
        return await run_tutorial_pipeline(
            request=request,
            user=user,
            session_id=str(body.session_id),
        )

    @router.post("/cancel", response_model=TutorialCancelResponse)
    async def cancel_tutorial(
        body: TutorialCancelRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> TutorialCancelResponse:
        return await cancel_tutorial_run(
            request=request,
            user=user,
            session_id=str(body.session_id),
        )

    @router.delete("/orphans", response_model=TutorialOrphanCleanupResponse)
    async def delete_tutorial_orphans(
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> TutorialOrphanCleanupResponse:
        return await cleanup_tutorial_orphans(request=request, user=user)

    return router
