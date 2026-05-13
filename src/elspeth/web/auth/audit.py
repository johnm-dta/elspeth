"""Request-bound web authentication audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fastapi import Request

from elspeth.contracts.auth import AuthProviderType
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

if TYPE_CHECKING:
    from elspeth.web.config import WebSettings


MAX_AUTH_AUDIT_TEXT_LENGTH = 512
"""Maximum length for caller-controlled auth audit context fields."""


class AuthAuditWriter(Protocol):
    """Interface consumed by auth routes for must-fire auth audit writes."""

    def record_login_success(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        user_id: str,
        username: str,
    ) -> None: ...

    def record_login_failure(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        username: str,
        failure_category: str,
    ) -> None: ...


def _bounded_text(value: str | None, *, max_length: int = MAX_AUTH_AUDIT_TEXT_LENGTH) -> str | None:
    if value is None:
        return None
    if len(value) <= max_length:
        return value
    return value[:max_length]


def _client_host(request: Request) -> str | None:
    client = request.client
    if client is None:
        return None
    return _bounded_text(client.host, max_length=128)


def _request_id(request: Request) -> str | None:
    request_id: str = request.state.request_id
    return _bounded_text(request_id, max_length=64)


def _optional_header(request: Request, name: str) -> str | None:
    if name not in request.headers:
        return None
    return request.headers[name]


def _request_metadata(request: Request) -> dict[str, object]:
    return {
        "method": request.method,
        "path": request.url.path,
    }


@dataclass(frozen=True)
class AuthAuditRecorder:
    """Synchronous Landscape writer for web authentication events."""

    landscape_url: str
    landscape_passphrase: str | None

    @classmethod
    def from_settings(cls, settings: WebSettings) -> AuthAuditRecorder:
        return cls(
            landscape_url=settings.get_landscape_url(),
            landscape_passphrase=settings.landscape_passphrase,
        )

    def record_login_success(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        user_id: str,
        username: str,
    ) -> None:
        with LandscapeDB.from_url(self.landscape_url, passphrase=self.landscape_passphrase) as db:
            RecorderFactory(db).auth_audit.record_login_outcome(
                outcome="success",
                provider=provider,
                user_id=user_id,
                username=username,
                failure_category=None,
                request_id=_request_id(request),
                client_host=_client_host(request),
                user_agent=_bounded_text(_optional_header(request, "user-agent")),
                metadata=_request_metadata(request),
            )

    def record_login_failure(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        username: str,
        failure_category: str,
    ) -> None:
        with LandscapeDB.from_url(self.landscape_url, passphrase=self.landscape_passphrase) as db:
            RecorderFactory(db).auth_audit.record_login_outcome(
                outcome="failure",
                provider=provider,
                user_id=None,
                username=username,
                failure_category=failure_category,
                request_id=_request_id(request),
                client_host=_client_host(request),
                user_agent=_bounded_text(_optional_header(request, "user-agent")),
                metadata=_request_metadata(request),
            )
