"""FastAPI router for the shareable-reviews endpoints.

Phase 6A Task 6 (UX redesign 2026-05). Three routes:

* ``POST /api/sessions/{session_id}/mark-ready-for-review``
* ``GET  /api/sessions/{session_id}/shareable-link``
* ``GET  /api/sessions/shared/{token}``

The first two require ``verify_session_ownership`` — same byte-identical
404 contract used by every other session-scoped route. The third is the
recipient's inspect view: it requires authentication (the token is a
capability, NOT an authenticator — see plan §"Capability vs authenticator"),
but does NOT call ownership verification because the recipient is
deliberately a different user from the session owner. Token-verification
failures map to 401; expired payload-store blobs map to 404.

Layer: L3 (application). Reads ``app.state.shareable_review_service``
which is wired in ``web/app.py``.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from elspeth.contracts.payload_store import PayloadNotFoundError
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.middleware.rate_limit import ComposerRateLimiter, get_rate_limiter
from elspeth.web.sessions.ownership import verify_session_ownership
from elspeth.web.shareable_reviews.models import (
    MarkReadyForReviewResponse,
    ShareableLinkResponse,
    SharedInspectResponse,
)
from elspeth.web.shareable_reviews.service import (
    CompositionNotRunnableError,
    ShareableReviewService,
)
from elspeth.web.shareable_reviews.signer import InvalidToken

_NO_STORE = "no-store"


def create_shareable_reviews_router() -> APIRouter:
    """Create the shareable-reviews router.

    Three endpoints; see module docstring for the access-control matrix.
    """
    router = APIRouter(tags=["shareable-reviews"])

    @router.post(
        "/api/sessions/{session_id}/mark-ready-for-review",
        response_model=MarkReadyForReviewResponse,
    )
    async def mark_ready_for_review(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    ) -> JSONResponse:
        """Mint a signed share artifact for the current composition state.

        Returns 409 with a generic detail if the composition fails either
        gate (validation or readiness-error). Returns 404 (byte-identical
        to "session does not exist") if the session belongs to a
        different user (IDOR). Returns 401 if the request is
        unauthenticated.
        """
        await rate_limiter.check(user.user_id)
        await verify_session_ownership(session_id, user, request)
        service: ShareableReviewService = request.app.state.shareable_review_service
        try:
            result = await service.mark_ready_for_review(session_id=session_id, user_id=user.user_id)
        except CompositionNotRunnableError as exc:
            # ``from exc``: preserves the server-side __context__ chain for
            # logs. Wire-facing ``detail`` is unaffected — only the internal
            # traceback chain. The probing-attacker rationale that justifies
            # ``from None`` on InvalidToken (below) does NOT apply here: a
            # 409 leaks no signal an authenticated session owner could not
            # already obtain by inspecting their own session.
            raise HTTPException(status_code=409, detail=exc.detail or exc.reason) from exc
        return JSONResponse(
            content=result.model_dump(mode="json"),
            headers={"Cache-Control": _NO_STORE},
        )

    @router.get(
        "/api/sessions/{session_id}/shareable-link",
        response_model=ShareableLinkResponse,
    )
    async def get_shareable_link(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    ) -> JSONResponse:
        """Re-mint a fresh token for the current (session, state).

        Content-addressing guarantees identical ``payload_digest`` across
        successive calls on an unchanged composition; only the token
        string differs (different nonce each call).
        """
        await rate_limiter.check(user.user_id)
        await verify_session_ownership(session_id, user, request)
        service: ShareableReviewService = request.app.state.shareable_review_service
        result = await service.get_shareable_link(session_id=session_id, user_id=user.user_id)
        return JSONResponse(
            content=result.model_dump(mode="json"),
            headers={"Cache-Control": _NO_STORE},
        )

    @router.get(
        "/api/sessions/shared/{token}",
        response_model=SharedInspectResponse,
    )
    async def get_shared_inspect(
        token: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
        rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),  # noqa: B008
    ) -> JSONResponse:
        """Read-only inspect view of a shared composition.

        The token is a CAPABILITY, not an authenticator — the recipient
        must still authenticate (``Depends(get_current_user)``). Without
        the authentication dependency, anyone with the URL gets in; that
        is NOT the designed behaviour. Do NOT "simplify" this dep away.

        ``InvalidToken`` (tampered/expired/malformed) → 401.
        ``PayloadNotFoundError`` (token verifies but the payload-store
        blob has been reaped under retention) → 404 with a message
        directing the recipient to request a fresh link.
        """
        await rate_limiter.check(user.user_id)
        service: ShareableReviewService = request.app.state.shareable_review_service
        try:
            result = await service.resolve_token(token=token, requesting_user_id=user.user_id)
        except InvalidToken:
            raise HTTPException(status_code=401, detail="Invalid or expired share token") from None
        except PayloadNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Shared snapshot is no longer available; ask the sender for a fresh link",
            ) from None
        return JSONResponse(
            content=result.model_dump(mode="json"),
            headers={"Cache-Control": _NO_STORE},
        )

    return router
