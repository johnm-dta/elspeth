"""FastAPI auth dependency -- extracts UserIdentity from Bearer tokens.

This is a FastAPI dependency function, not ASGI middleware. All protected
routes declare it via Depends(get_current_user).
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from elspeth.web.auth.models import AuthenticationError, UserIdentity
from elspeth.web.auth.protocol import AuthProvider


async def get_current_user(request: Request) -> UserIdentity:
    """Extract and validate a Bearer token from the request.

    Retrieves the auth_provider from request.app.state and calls
    authenticate(token). Converts AuthenticationError to HTTP 401.

    Stashes the raw token on request.state.auth_token so downstream
    route handlers (e.g. /me) can reuse it without re-parsing the
    Authorization header.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
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
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=exc.detail) from exc
