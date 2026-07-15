"""Request-bound web authentication audit helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, cast

import jwt as pyjwt
import structlog
from fastapi import Request
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.auth import AuthProviderType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.web.auth.models import AuthenticationError, AuthProviderUnavailable
from elspeth.web.landscape_access import landscape_create_tables_allowed

if TYPE_CHECKING:
    from elspeth.web.config import WebSettings


_slog = structlog.get_logger(__name__)


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

    def record_token_issued(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        user_id: str,
        username: str,
        access_token: str,
        issuance_path: str,
    ) -> None: ...

    def record_auth_failure(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        failure_category: str,
        failure_stage: str,
        user_id: str | None,
        username: str | None,
        exception_class: str | None,
    ) -> None: ...


class AuthAuditOperation(StrEnum):
    LOGIN_SUCCESS = "login_success"
    TOKEN_ISSUED = "token_issued"
    AUTH_FAILURE = "auth_failure"
    LOGIN_FAILURE = "login_failure"


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


def _issued_token_claims(access_token: str) -> dict[str, object]:
    try:
        decoded = pyjwt.decode(access_token, options={"verify_signature": False})
    except pyjwt.PyJWTError as exc:
        raise AuditIntegrityError("Issued access token could not be decoded for auth audit metadata") from exc
    return cast(dict[str, object], decoded)


def _required_int_claim(claims: dict[str, object], claim_name: str) -> int:
    if claim_name not in claims:
        raise AuditIntegrityError(f"Issued access token missing {claim_name!r} claim for auth audit metadata")
    value = claims[claim_name]
    if type(value) is not int:
        raise AuditIntegrityError(f"Issued access token {claim_name!r} claim must be int for auth audit metadata")
    return value


def _token_issued_metadata(request: Request, *, access_token: str, issuance_path: str) -> dict[str, object]:
    claims = _issued_token_claims(access_token)
    metadata = _request_metadata(request)
    metadata["issuance_path"] = issuance_path
    metadata["token_type"] = "bearer"
    metadata["issued_at"] = _required_int_claim(claims, "iat")
    metadata["expires_at"] = _required_int_claim(claims, "exp")
    return metadata


def classify_authentication_failure(exc: AuthenticationError) -> str:
    """Classify auth errors without storing their external-data-bearing detail."""
    if type(exc) is AuthProviderUnavailable:
        return "provider_unavailable"

    detail = exc.detail
    if detail.startswith("Invalid tenant") or detail.startswith("Missing tenant claim"):
        return "tenant_claim_invalid"
    if detail.startswith("Missing required") or "group overage marker" in detail or detail.startswith("OIDC profile claim"):
        return "claims_invalid"
    if (
        detail.startswith("Invalid token")
        or detail.startswith("Token header")
        or detail.startswith("No matching key")
        or detail.startswith("JWKS key")
    ):
        return "invalid_token"
    if detail.startswith("JWKS document") or detail.startswith("OIDC discovery document"):
        return "provider_metadata_invalid"
    return "authentication_error"


@dataclass(frozen=True)
class AuthAuditRecorder:
    """Synchronous Landscape writer for web authentication events."""

    landscape_url: str
    landscape_passphrase: str | None
    create_tables: bool

    @classmethod
    def from_settings(cls, settings: WebSettings) -> AuthAuditRecorder:
        return cls(
            landscape_url=settings.get_landscape_url(),
            landscape_passphrase=settings.landscape_passphrase,
            create_tables=landscape_create_tables_allowed(settings),
        )

    @contextmanager
    def _open_landscape(self, operation: AuthAuditOperation) -> Iterator[LandscapeDB]:
        try:
            with LandscapeDB.from_url(
                self.landscape_url,
                passphrase=self.landscape_passphrase,
                create_tables=self.create_tables,
            ) as db:
                yield db
        except (SchemaCompatibilityError, LandscapeRecordError, SQLAlchemyError, OSError) as exc:
            _slog.error(
                "auth_audit_write_failed",
                operation=operation,
                exception_class=type(exc).__name__,
            )
            raise

    def record_login_success(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        user_id: str,
        username: str,
    ) -> None:
        with self._open_landscape(AuthAuditOperation.LOGIN_SUCCESS) as db:
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

    def record_token_issued(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        user_id: str,
        username: str,
        access_token: str,
        issuance_path: str,
    ) -> None:
        with self._open_landscape(AuthAuditOperation.TOKEN_ISSUED) as db:
            RecorderFactory(db).auth_audit.record_token_issued(
                provider=provider,
                user_id=user_id,
                username=username,
                request_id=_request_id(request),
                client_host=_client_host(request),
                user_agent=_bounded_text(_optional_header(request, "user-agent")),
                metadata=_token_issued_metadata(
                    request,
                    access_token=access_token,
                    issuance_path=issuance_path,
                ),
            )

    def record_auth_failure(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        failure_category: str,
        failure_stage: str,
        user_id: str | None,
        username: str | None,
        exception_class: str | None,
    ) -> None:
        metadata = _request_metadata(request)
        metadata["failure_stage"] = failure_stage
        metadata["exception_class"] = exception_class
        with self._open_landscape(AuthAuditOperation.AUTH_FAILURE) as db:
            RecorderFactory(db).auth_audit.record_auth_failure(
                provider=provider,
                user_id=user_id,
                username=username,
                failure_category=failure_category,
                request_id=_request_id(request),
                client_host=_client_host(request),
                user_agent=_bounded_text(_optional_header(request, "user-agent")),
                metadata=metadata,
            )

    def record_login_failure(
        self,
        request: Request,
        *,
        provider: AuthProviderType,
        username: str,
        failure_category: str,
    ) -> None:
        with self._open_landscape(AuthAuditOperation.LOGIN_FAILURE) as db:
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
