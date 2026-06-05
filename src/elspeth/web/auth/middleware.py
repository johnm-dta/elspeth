"""FastAPI auth dependency -- extracts UserIdentity from Bearer tokens.

This is a FastAPI dependency function, not ASGI middleware. All protected
routes declare it via Depends(get_current_user).
"""

from __future__ import annotations

from typing import cast

from fastapi import HTTPException, Request

from elspeth.contracts.auth import AuthProviderType
from elspeth.web.auth.audit import AuthAuditWriter, classify_authentication_failure
from elspeth.web.auth.models import AuthenticationError, AuthProviderUnavailable, UserIdentity
from elspeth.web.auth.protocol import AuthProvider
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import check_auth_rate_limit


def _auth_audit_recorder(request: Request) -> AuthAuditWriter:
    return cast(AuthAuditWriter, request.app.state.auth_audit_recorder)


def _settings(request: Request) -> WebSettings:
    return cast(WebSettings, request.app.state.settings)


async def _record_auth_failure_after_rate_limit(
    request: Request,
    *,
    provider: AuthProviderType,
    failure_category: str,
    failure_stage: str,
    user_id: str | None,
    username: str | None,
    exception_class: str | None,
) -> None:
    """Rate-limit attacker-triggerable auth failure audit writes.

    ``get_current_user`` is shared by protected API routes, so missing,
    malformed, or invalid Authorization headers are reachable before a caller
    has authenticated. Reuse the auth endpoint's per-client limiter before the
    durable audit write so repeated unauthenticated failures cannot force
    unbounded synchronous database work. If the limiter raises 429, the audit
    write for that over-limit attempt is intentionally skipped.
    """
    await check_auth_rate_limit(request)
    _auth_audit_recorder(request).record_auth_failure(
        request,
        provider=provider,
        failure_category=failure_category,
        failure_stage=failure_stage,
        user_id=user_id,
        username=username,
        exception_class=exception_class,
    )


async def get_current_user(request: Request) -> UserIdentity:
    """Extract and validate a Bearer token from the request.

    Retrieves the auth_provider from request.app.state and calls
    authenticate(token). Converts AuthProviderUnavailable to HTTP 503 and
    AuthenticationError to HTTP 401.

    Stashes the raw token on request.state.auth_token so downstream
    route handlers (e.g. /me) can reuse it without re-parsing the
    Authorization header.
    """
    settings = _settings(request)

    if "Authorization" not in request.headers:
        await _record_auth_failure_after_rate_limit(
            request,
            provider=settings.auth_provider,
            failure_category="missing_authorization_header",
            failure_stage="authorization_header",
            user_id=None,
            username=None,
            exception_class=None,
        )
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )
    auth_header = request.headers["Authorization"]

    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        await _record_auth_failure_after_rate_limit(
            request,
            provider=settings.auth_provider,
            failure_category="invalid_authorization_header",
            failure_stage="authorization_header",
            user_id=None,
            username=None,
            exception_class=None,
        )
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    token = parts[1].strip()
    request.state.auth_token = token

    # Decode claims without verification for downstream use (e.g. iat
    # for refresh chain enforcement).  The authenticated call below
    # verifies the signature — this is a pure parse, not a trust decision.
    # On decode failure, set None (not {}) so downstream can distinguish
    # "no iat in valid claims" from "claims unparseable."
    import jwt as _jwt

    try:
        request.state.auth_claims = _jwt.decode(token, options={"verify_signature": False}, algorithms=["HS256"])
    except _jwt.PyJWTError:
        request.state.auth_claims = None

    auth_provider: AuthProvider = request.app.state.auth_provider

    try:
        return await auth_provider.authenticate(token)
    except AuthProviderUnavailable as exc:
        await _record_auth_failure_after_rate_limit(
            request,
            provider=settings.auth_provider,
            failure_category="provider_unavailable",
            failure_stage="authenticate",
            user_id=None,
            username=None,
            exception_class=type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail=exc.detail) from exc
    except AuthenticationError as exc:
        await _record_auth_failure_after_rate_limit(
            request,
            provider=settings.auth_provider,
            failure_category=classify_authentication_failure(exc),
            failure_stage="authenticate",
            user_id=None,
            username=None,
            exception_class=type(exc).__name__,
        )
        raise HTTPException(status_code=401, detail=exc.detail) from exc
