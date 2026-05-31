"""Shared session-ownership verification for route handlers.

L3 peer module. ``execution/routes.py`` and ``audit_readiness/routes.py``
both verify that a session_id path-param resolves to a session owned by
the authenticated user. The verb is identical in both call sites — a
single source of truth here prevents drift between the two IDOR checks.

IDOR contract: every access-control failure (session does not exist,
session owned by a different user, session bound to a different
auth_provider_type) returns ``404 Session not found``. Returning 403 or
distinguishable details would let an authenticated attacker probe
arbitrary UUIDs and learn which ones exist in OTHER users' workspaces.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, Request

from elspeth.web.auth.models import UserIdentity
from elspeth.web.config import WebSettings
from elspeth.web.sessions.protocol import SessionNotFoundError, SessionServiceProtocol


async def verify_session_ownership(
    session_id: UUID,
    user: UserIdentity,
    request: Request,
) -> None:
    """Verify the session exists and belongs to the current user.

    Returns 404 (not 403) to avoid leaking session existence (IDOR).
    Reads ``session_service`` and ``settings`` from ``request.app.state``.

    Raises:
        HTTPException(404): session does not exist, session user_id
            differs from ``user.user_id``, or session
            ``auth_provider_type`` differs from
            ``settings.auth_provider``.
    """
    session_service: SessionServiceProtocol = request.app.state.session_service
    settings: WebSettings = request.app.state.settings
    try:
        session = await session_service.get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found") from None

    if session.user_id != user.user_id or session.auth_provider_type != settings.auth_provider:
        raise HTTPException(status_code=404, detail="Session not found")
