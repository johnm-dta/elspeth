"""Tests for EntraAuthProvider -- tenant validation and group claim extraction."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elspeth.web.auth.entra import EntraAuthProvider
from elspeth.web.auth.models import AuthenticationError
from tests.unit.web.auth.conftest import make_rs256_token

TENANT_ID = "00000000-aaaa-bbbb-cccc-111111111111"
AUDIENCE = "my-entra-app-id"
ISSUER = f"https://login.microsoftonline.com/{TENANT_ID}/v2.0"


def _valid_entra_claims(overrides: dict[str, object] | None = None) -> dict[str, object]:
    """Return valid Entra ID JWT claims with optional overrides."""
    claims = {
        "sub": "entra-user-456",
        "preferred_username": "alice@contoso.com",
        "name": "Alice Contoso",
        "email": "alice@contoso.com",
        "tid": TENANT_ID,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 3600,
    }
    if overrides:
        claims.update(overrides)
    return claims


@pytest.fixture
def mock_httpx_discovery(jwks_response):
    """Patch httpx.AsyncClient to return OIDC discovery and JWKS responses.

    NOTE: Similar fixture exists in test_oidc_provider.py.
    Intentionally kept separate — different ISSUER and JWKS URL patterns.
    """

    async def mock_get(url, **kwargs):
        response = MagicMock()
        response.raise_for_status = lambda: None
        if ".well-known/openid-configuration" in url:
            response.json.return_value = {
                "jwks_uri": f"{ISSUER}/discovery/v2.0/keys",
                "issuer": ISSUER,
            }
        elif "keys" in url:
            response.json.return_value = jwks_response
        return response

    client_mock = AsyncMock()
    client_mock.get = mock_get
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=False)

    return patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=client_mock)


class TestEntraTenantValidation:
    """Tests for tenant ID validation."""

    @pytest.mark.asyncio
    async def test_valid_tenant_passes(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_entra_claims())
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
        assert identity.user_id == "entra-user-456"
        assert identity.username == "alice@contoso.com"

    @pytest.mark.asyncio
    async def test_wrong_tenant_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        claims = _valid_entra_claims(
            {
                "tid": "wrong-tenant-id",
                "iss": ISSUER,
            }
        )
        provider_correct = EntraAuthProvider(
            tenant_id=TENANT_ID,
            audience=AUDIENCE,
        )
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery, pytest.raises(AuthenticationError, match="Invalid tenant"):
            await provider_correct.authenticate(token)

    @pytest.mark.asyncio
    async def test_missing_tid_claim_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Token without a tid claim should fail with a specific message."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims()
        del claims["tid"]
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery, pytest.raises(AuthenticationError, match="Missing tenant claim"):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_missing_sub_claim_raises_authentication_error(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Token without sub claim must raise AuthenticationError, not KeyError."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims()
        del claims["sub"]
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery, pytest.raises(AuthenticationError, match="Missing required 'sub' claim"):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_blank_sub_claim_raises_authentication_error(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Token with blank sub must raise AuthenticationError."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_entra_claims({"sub": ""}))
        with mock_httpx_discovery, pytest.raises(AuthenticationError, match="user_id"):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_null_preferred_username_authenticate_falls_back_to_sub(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """authenticate() with null preferred_username falls back to sub."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_entra_claims({"preferred_username": None}))
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
        assert identity.username == "entra-user-456"

    @pytest.mark.asyncio
    async def test_empty_preferred_username_authenticate_falls_back_to_sub(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """authenticate() with empty preferred_username falls back to sub."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_entra_claims({"preferred_username": ""}))
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
        assert identity.username == "entra-user-456"


class TestEntraGroupClaims:
    """Tests for group and role claim extraction."""

    @pytest.mark.asyncio
    async def test_group_ids_extracted_to_profile(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        group_ids = [
            "11111111-aaaa-0000-0000-000000000001",
            "22222222-bbbb-0000-0000-000000000002",
        ]
        claims = _valid_entra_claims({"groups": group_ids})
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert "11111111-aaaa-0000-0000-000000000001" in profile.groups
        assert "22222222-bbbb-0000-0000-000000000002" in profile.groups

    @pytest.mark.asyncio
    async def test_roles_prefixed_in_groups(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims(
            {
                "groups": ["group-1"],
                "roles": ["admin", "reader"],
            }
        )
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert "group-1" in profile.groups
        assert "role:admin" in profile.groups
        assert "role:reader" in profile.groups

    @pytest.mark.asyncio
    async def test_no_groups_or_roles(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims()
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.groups == ()

    @pytest.mark.asyncio
    async def test_no_name_claims_returns_none_display_name(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """When IdP provides neither name nor preferred_username, display_name is None."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims()
        del claims["name"]
        del claims["preferred_username"]
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.display_name is None
        assert profile.username == "entra-user-456"  # Falls back to sub

    @pytest.mark.asyncio
    async def test_non_list_groups_claim_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """A groups claim that is not a list should raise AuthenticationError."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims({"groups": "not-a-list"})
        token = make_rs256_token(private_key, claims)
        with (
            mock_httpx_discovery,
            pytest.raises(
                AuthenticationError,
                match="Unexpected type for 'groups' claim",
            ),
        ):
            await provider.get_user_info(token)

    @pytest.mark.asyncio
    async def test_non_list_roles_claim_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """A roles claim that is not a list should raise AuthenticationError."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims({"groups": ["valid-group"], "roles": 42})
        token = make_rs256_token(private_key, claims)
        with (
            mock_httpx_discovery,
            pytest.raises(
                AuthenticationError,
                match="Unexpected type for 'roles' claim",
            ),
        ):
            await provider.get_user_info(token)


class TestEntraGetUserInfoTenantValidation:
    """Tests for tenant validation in get_user_info (not just authenticate)."""

    @pytest.mark.asyncio
    async def test_get_user_info_wrong_tenant_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """get_user_info must also validate the tenant, not just authenticate."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims({"tid": "wrong-tenant-id"})
        token = make_rs256_token(private_key, claims)
        with (
            mock_httpx_discovery,
            pytest.raises(
                AuthenticationError,
                match="Invalid tenant",
            ),
        ):
            await provider.get_user_info(token)

    @pytest.mark.asyncio
    async def test_get_user_info_missing_tid_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """get_user_info with no tid claim should raise AuthenticationError."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims()
        del claims["tid"]
        token = make_rs256_token(private_key, claims)
        with (
            mock_httpx_discovery,
            pytest.raises(
                AuthenticationError,
                match="Missing tenant claim",
            ),
        ):
            await provider.get_user_info(token)

    @pytest.mark.asyncio
    async def test_get_user_info_missing_sub_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """get_user_info must reject tokens without sub claim."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        claims = _valid_entra_claims()
        del claims["sub"]
        token = make_rs256_token(private_key, claims)
        with (
            mock_httpx_discovery,
            pytest.raises(
                AuthenticationError,
                match="Missing required 'sub' claim",
            ),
        ):
            await provider.get_user_info(token)

    @pytest.mark.asyncio
    async def test_get_user_info_blank_sub_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """get_user_info with blank sub must raise AuthenticationError."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_entra_claims({"sub": ""}))
        with mock_httpx_discovery, pytest.raises(AuthenticationError, match="user_id"):
            await provider.get_user_info(token)

    @pytest.mark.asyncio
    async def test_get_user_info_null_preferred_username_falls_back_to_sub(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Entra may send preferred_username: null — must fall back to sub."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_entra_claims({"preferred_username": None}))
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.username == "entra-user-456"

    @pytest.mark.asyncio
    async def test_get_user_info_empty_preferred_username_falls_back_to_sub(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Entra with empty preferred_username falls back to sub."""
        private_key, _ = rsa_keypair
        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_entra_claims({"preferred_username": ""}))
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.username == "entra-user-456"


class TestEntraProtocolConformance:
    """Verify EntraAuthProvider satisfies the AuthProvider protocol."""

    def test_entra_satisfies_auth_provider(self) -> None:
        from elspeth.web.auth.protocol import AuthProvider

        provider = EntraAuthProvider(tenant_id=TENANT_ID, audience=AUDIENCE)
        assert isinstance(provider, AuthProvider)
