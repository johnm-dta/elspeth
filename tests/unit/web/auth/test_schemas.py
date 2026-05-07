"""Tier 1 strictness regression tests for auth response schemas.

Auth responses expose identity, group membership, and token material —
fields that the audit trail and security boundary both rely on.  The
response models inherit from ``_StrictResponse`` so that:

* An accidental ``groups=None`` (instead of ``[]``) crashes at
  construction rather than reaching clients with a partially-decoded
  identity.
* An extra field introduced by a buggy provider adapter cannot smuggle
  unauthorized claims into the ``/me`` or ``/config`` surface.

Request models (``LoginRequest``, ``RegisterRequest``) remain
plain ``BaseModel`` — Tier 3 input with its own field validators.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.web.auth.routes import (
    AuthConfigResponse,
    TokenResponse,
    UserProfileResponse,
)


class TestAuthStrictCoercionRejected:
    def test_token_response_rejects_int_for_access_token(self) -> None:
        with pytest.raises(ValidationError):
            TokenResponse(access_token=12345)  # type: ignore[arg-type]

    def test_user_profile_rejects_int_for_user_id(self) -> None:
        with pytest.raises(ValidationError):
            UserProfileResponse(user_id=42, username="u")  # type: ignore[arg-type]

    def test_user_profile_rejects_non_list_groups(self) -> None:
        with pytest.raises(ValidationError):
            UserProfileResponse(user_id="u1", username="u", groups="admins")  # type: ignore[arg-type]

    def test_auth_config_rejects_int_for_provider(self) -> None:
        with pytest.raises(ValidationError):
            AuthConfigResponse(provider=1, registration_mode="open")  # type: ignore[arg-type]


class TestAuthExtraFieldsRejected:
    def test_token_response_rejects_extra(self) -> None:
        """``refresh_token`` is deliberately not part of the response —
        the API is bearer-only.  Adding it silently would be a security
        regression."""
        with pytest.raises(ValidationError, match="extra"):
            TokenResponse(access_token="jwt", refresh_token="r")  # type: ignore[call-arg]

    def test_user_profile_rejects_extra(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            UserProfileResponse(
                user_id="u1",
                username="u",
                display_name=None,
                email=None,
                groups=[],
                admin=True,  # type: ignore[call-arg]
            )

    def test_auth_config_rejects_extra(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            AuthConfigResponse(
                provider="local",
                registration_mode="open",
                client_secret="shhh",  # type: ignore[call-arg]
            )


class TestAuthHappyPath:
    def test_token_response_defaults_to_bearer(self) -> None:
        resp = TokenResponse(access_token="jwt")
        assert resp.token_type == "bearer"

    def test_user_profile_with_optional_fields_unset(self) -> None:
        resp = UserProfileResponse(user_id="u1", username="u")
        assert resp.display_name is None
        assert resp.email is None
        assert resp.groups == []

    def test_user_profile_with_groups(self) -> None:
        resp = UserProfileResponse(
            user_id="u1",
            username="u",
            groups=["admin", "auditor"],
        )
        assert resp.groups == ["admin", "auditor"]

    def test_auth_config_local_defaults(self) -> None:
        resp = AuthConfigResponse(provider="local", registration_mode="open")
        assert resp.oidc_issuer is None
        assert resp.authorization_endpoint is None
