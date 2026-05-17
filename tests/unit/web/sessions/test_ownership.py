"""Tests for shared session ownership verification."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException

from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions import protocol
from elspeth.web.sessions.ownership import verify_session_ownership
from elspeth.web.sessions.protocol import SessionRecord

_SESSION_ID = UUID("11111111-1111-1111-1111-111111111111")
_USER = UserIdentity(user_id="alice", username="alice")


class _Settings:
    auth_provider = "local"


def _request(session_service: object) -> object:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                session_service=session_service,
                settings=_Settings(),
            )
        )
    )


class _MissingSessionService:
    async def get_session(self, session_id: UUID) -> SessionRecord:
        not_found_error = getattr(protocol, "SessionNotFoundError", ValueError)
        raise not_found_error(session_id)


class _BuggySessionService:
    async def get_session(self, session_id: UUID) -> SessionRecord:
        raise ValueError("UUID coercion broke while constructing SessionRecord")


@pytest.mark.asyncio
async def test_missing_session_maps_to_idor_safe_404() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await verify_session_ownership(_SESSION_ID, _USER, _request(_MissingSessionService()))  # type: ignore[arg-type]

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Session not found"


@pytest.mark.asyncio
async def test_unrelated_value_error_propagates() -> None:
    with pytest.raises(ValueError, match="UUID coercion broke"):
        await verify_session_ownership(_SESSION_ID, _USER, _request(_BuggySessionService()))  # type: ignore[arg-type]
