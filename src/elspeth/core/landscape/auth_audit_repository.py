"""Web authentication audit repository for Landscape."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from elspeth.contracts.auth import AuthProviderType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import generate_id, now
from elspeth.core.landscape.schema import auth_events_table

AuthAuditEventType = Literal["login", "token_issued", "auth_failure"]
AuthAuditOutcome = Literal["success", "failure"]

AUTH_AUDIT_EVENT_TYPES: tuple[AuthAuditEventType, ...] = ("login", "token_issued", "auth_failure")
AUTH_AUDIT_SUCCESS: AuthAuditOutcome = "success"
AUTH_AUDIT_FAILURE: AuthAuditOutcome = "failure"
AUTH_AUDIT_OUTCOMES: tuple[AuthAuditOutcome, ...] = (AUTH_AUDIT_SUCCESS, AUTH_AUDIT_FAILURE)


class AuthAuditRepository:
    """Record non-run-scoped web authentication events in Landscape."""

    def __init__(self, ops: DatabaseOps) -> None:
        self._ops = ops

    def record_auth_event(
        self,
        *,
        event_type: AuthAuditEventType,
        outcome: AuthAuditOutcome,
        provider: AuthProviderType,
        user_id: str | None,
        username: str | None,
        failure_category: str | None,
        request_id: str | None,
        client_host: str | None,
        user_agent: str | None,
        metadata: Mapping[str, object],
    ) -> str:
        """Record an auth event synchronously before the HTTP response is sent."""
        if event_type not in AUTH_AUDIT_EVENT_TYPES:
            raise AuditIntegrityError(f"Unsupported auth audit event_type: {event_type!r}")
        if outcome not in AUTH_AUDIT_OUTCOMES:
            raise AuditIntegrityError(f"Unsupported auth audit outcome: {outcome!r}")
        if outcome == AUTH_AUDIT_SUCCESS and failure_category is not None:
            raise AuditIntegrityError("Successful auth audit events must not carry failure_category")
        if outcome == AUTH_AUDIT_FAILURE and failure_category is None:
            raise AuditIntegrityError("Failed auth audit events must carry failure_category")

        event_id = generate_id()
        self._ops.execute_insert(
            auth_events_table.insert().values(
                event_id=event_id,
                occurred_at=now(),
                event_type=event_type,
                outcome=outcome,
                provider=provider,
                user_id=user_id,
                username=username,
                failure_category=failure_category,
                request_id=request_id,
                client_host=client_host,
                user_agent=user_agent,
                metadata_json=canonical_json(metadata),
            ),
            context=f"record_auth_event event_type={event_type} outcome={outcome}",
        )
        return event_id

    def record_login_outcome(
        self,
        *,
        outcome: AuthAuditOutcome,
        provider: AuthProviderType,
        user_id: str | None,
        username: str | None,
        failure_category: str | None,
        request_id: str | None,
        client_host: str | None,
        user_agent: str | None,
        metadata: Mapping[str, object],
    ) -> str:
        """Record a local login success or failed credential attempt."""
        return self.record_auth_event(
            event_type="login",
            outcome=outcome,
            provider=provider,
            user_id=user_id,
            username=username,
            failure_category=failure_category,
            request_id=request_id,
            client_host=client_host,
            user_agent=user_agent,
            metadata=metadata,
        )

    def record_token_issued(
        self,
        *,
        provider: AuthProviderType,
        user_id: str,
        username: str,
        request_id: str | None,
        client_host: str | None,
        user_agent: str | None,
        metadata: Mapping[str, object],
    ) -> str:
        """Record access-token issuance without storing the bearer token."""
        return self.record_auth_event(
            event_type="token_issued",
            outcome=AUTH_AUDIT_SUCCESS,
            provider=provider,
            user_id=user_id,
            username=username,
            failure_category=None,
            request_id=request_id,
            client_host=client_host,
            user_agent=user_agent,
            metadata=metadata,
        )

    def record_auth_failure(
        self,
        *,
        provider: AuthProviderType,
        user_id: str | None,
        username: str | None,
        failure_category: str,
        request_id: str | None,
        client_host: str | None,
        user_agent: str | None,
        metadata: Mapping[str, object],
    ) -> str:
        """Record an authentication or profile-lookup failure classification."""
        return self.record_auth_event(
            event_type="auth_failure",
            outcome=AUTH_AUDIT_FAILURE,
            provider=provider,
            user_id=user_id,
            username=username,
            failure_category=failure_category,
            request_id=request_id,
            client_host=client_host,
            user_agent=user_agent,
            metadata=metadata,
        )
