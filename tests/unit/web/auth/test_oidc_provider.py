"""Tests for OIDCAuthProvider -- JWKS discovery, token validation."""

from __future__ import annotations

import asyncio
import inspect
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from elspeth.web.auth.models import AuthenticationError, UserIdentity
from elspeth.web.auth.oidc import OIDCAuthProvider
from tests.unit.web.auth.conftest import build_rsa_jwk, make_rs256_token, make_rsa_token

ISSUER = "https://login.example.com"
AUDIENCE = "my-app-client-id"


def _valid_claims(overrides: dict[str, object] | None = None) -> dict[str, object]:
    """Return a valid JWT claims dict with optional overrides."""
    claims = {
        "sub": "user-123",
        "preferred_username": "alice",
        "name": "Alice Smith",
        "email": "alice@example.com",
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

    NOTE: Similar fixture exists in test_entra_provider.py.
    Intentionally kept separate — different ISSUER and JWKS URL patterns.
    """

    async def mock_get(url, **kwargs):
        response = MagicMock()
        response.raise_for_status = lambda: None
        if ".well-known/openid-configuration" in url:
            response.json.return_value = {
                "jwks_uri": f"{ISSUER}/keys",
                "issuer": ISSUER,
            }
        elif url.endswith("/keys"):
            response.json.return_value = jwks_response
        return response

    client_mock = AsyncMock()
    client_mock.get = mock_get
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=False)

    return patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=client_mock)


class TestOIDCDiscovery:
    """Tests for JWKS discovery and caching."""

    @pytest.mark.asyncio
    async def test_fetches_jwks_on_first_authenticate(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims())
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
            # If authenticate succeeds, JWKS discovery must have been called
            # (the provider has no cached keys on first call)
            assert identity.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_caches_jwks_on_subsequent_calls(
        self,
        rsa_keypair,
        jwks_response,
    ) -> None:
        """Second authenticate should use cached JWKS, not re-fetch."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=3600,
        )
        token = make_rs256_token(private_key, _valid_claims())

        # First call: mock returns valid JWKS
        async def success_get(url, **kwargs):
            response = MagicMock()
            response.raise_for_status = lambda: None
            if ".well-known/openid-configuration" in url:
                response.json.return_value = {"jwks_uri": f"{ISSUER}/keys", "issuer": ISSUER}
            elif url.endswith("/keys"):
                response.json.return_value = jwks_response
            return response

        success_client = AsyncMock()
        success_client.get = success_get
        success_client.__aenter__ = AsyncMock(return_value=success_client)
        success_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=success_client):
            await provider.authenticate(token)

        # Second call: mock raises on any HTTP call -- if caching works,
        # this mock is never hit
        async def failing_get(url, **kwargs):
            raise AssertionError("JWKS should have been cached -- HTTP call should not happen")

        failing_client = AsyncMock()
        failing_client.get = failing_get
        failing_client.__aenter__ = AsyncMock(return_value=failing_client)
        failing_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=failing_client):
            identity = await provider.authenticate(token)
            assert identity.user_id == "user-123"


class TestOIDCTokenValidation:
    """Tests for token validation checks."""

    @pytest.mark.asyncio
    async def test_valid_token_returns_identity(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims())
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
        assert isinstance(identity, UserIdentity)
        assert identity.user_id == "user-123"
        assert identity.username == "alice"

    @pytest.mark.asyncio
    async def test_wrong_audience_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience="wrong-audience")
        token = make_rs256_token(private_key, _valid_claims())
        with mock_httpx_discovery, pytest.raises(AuthenticationError):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_wrong_issuer_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer="https://wrong-issuer.com",
            audience=AUDIENCE,
        )
        token = make_rs256_token(private_key, _valid_claims())
        with mock_httpx_discovery, pytest.raises(AuthenticationError):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_expired_token_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims({"exp": int(time.time()) - 10})
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery, pytest.raises(AuthenticationError):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_missing_sub_claim_raises_authentication_error(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Token without sub claim must raise AuthenticationError, not KeyError."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims()
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
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims({"sub": ""}))
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
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims({"preferred_username": None}))
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
        assert identity.username == "user-123"

    @pytest.mark.asyncio
    async def test_empty_preferred_username_authenticate_falls_back_to_sub(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """authenticate() with empty preferred_username falls back to sub."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims({"preferred_username": ""}))
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
        assert identity.username == "user-123"


class TestOIDCGetUserInfo:
    """Tests for full profile retrieval from OIDC claims."""

    @pytest.mark.asyncio
    async def test_get_user_info_returns_profile(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims({"groups": ["team-a", "team-b"]})
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.user_id == "user-123"
        assert profile.display_name == "Alice Smith"
        assert profile.email == "alice@example.com"
        assert profile.groups == ("team-a", "team-b")

    @pytest.mark.asyncio
    async def test_get_user_info_no_optional_claims(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims()
        # Remove optional fields
        del claims["email"]
        del claims["name"]
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.display_name == "alice"  # Falls back to preferred_username
        assert profile.email is None
        assert profile.groups == ()

    @pytest.mark.asyncio
    async def test_get_user_info_no_name_claims_returns_none(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """When IdP provides neither name nor preferred_username, display_name is None."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims()
        del claims["name"]
        del claims["preferred_username"]
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.display_name is None
        assert profile.username == "user-123"  # Falls back to sub

    @pytest.mark.asyncio
    async def test_non_list_groups_claim_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """A groups claim that is not a list should raise AuthenticationError."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims({"groups": "not-a-list"})
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
    async def test_authenticate_missing_preferred_username(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Without preferred_username, username should fall back to sub."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims()
        del claims["preferred_username"]
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
        assert identity.username == identity.user_id
        assert identity.username == "user-123"

    @pytest.mark.asyncio
    async def test_get_user_info_missing_sub_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """get_user_info must also reject tokens without sub claim."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        claims = _valid_claims()
        del claims["sub"]
        token = make_rs256_token(private_key, claims)
        with mock_httpx_discovery, pytest.raises(AuthenticationError, match="Missing required 'sub' claim"):
            await provider.get_user_info(token)

    @pytest.mark.asyncio
    async def test_get_user_info_blank_sub_raises(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """get_user_info with blank sub must raise AuthenticationError."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims({"sub": ""}))
        with mock_httpx_discovery, pytest.raises(AuthenticationError, match="user_id"):
            await provider.get_user_info(token)

    @pytest.mark.asyncio
    async def test_null_preferred_username_falls_back_to_sub(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """IdP may send preferred_username: null — username must fall back to sub."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims({"preferred_username": None}))
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.username == "user-123"

    @pytest.mark.asyncio
    async def test_empty_preferred_username_falls_back_to_sub(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """IdP may send preferred_username: "" — username must fall back to sub."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims({"preferred_username": ""}))
        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)
        assert profile.username == "user-123"

    @pytest.mark.asyncio
    async def test_non_string_optional_profile_claims_are_dropped(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Cosmetic IdP metadata must not crash profile construction when it has the wrong type."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims({"name": 42, "email": 123}))

        with mock_httpx_discovery:
            profile = await provider.get_user_info(token)

        assert profile.user_id == "user-123"
        assert profile.username == "alice"
        assert profile.display_name == "alice"
        assert profile.email is None


class TestOIDCJWKSFailures:
    """Tests for JWKS discovery network failures."""

    @pytest.mark.asyncio
    async def test_jwks_connect_error_raises(self) -> None:
        """httpx.ConnectError during discovery raises AuthenticationError."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)

        async def mock_get(url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        client_mock = AsyncMock()
        client_mock.get = mock_get
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "elspeth.web.auth.oidc.httpx.AsyncClient",
                return_value=client_mock,
            ),
            pytest.raises(AuthenticationError, match="JWKS unavailable"),
        ):
            await provider.authenticate("some-token")

    @pytest.mark.asyncio
    async def test_jwks_missing_jwks_uri_key_raises(self) -> None:
        """Discovery JSON without jwks_uri key raises AuthenticationError."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)

        async def mock_get(url, **kwargs):
            response = MagicMock()
            response.raise_for_status = lambda: None
            # Discovery response missing the jwks_uri key
            response.json.return_value = {"issuer": ISSUER}
            return response

        client_mock = AsyncMock()
        client_mock.get = mock_get
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "elspeth.web.auth.oidc.httpx.AsyncClient",
                return_value=client_mock,
            ),
            pytest.raises(
                AuthenticationError,
                match="missing non-empty string 'jwks_uri'",
            ),
        ):
            await provider.authenticate("some-token")

    @pytest.mark.asyncio
    async def test_jwks_malformed_json_raises(self) -> None:
        """Discovery response with malformed JSON raises AuthenticationError."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)

        async def mock_get(url, **kwargs):
            response = MagicMock()
            response.raise_for_status = lambda: None
            response.json.side_effect = ValueError("No JSON object could be decoded")
            return response

        client_mock = AsyncMock()
        client_mock.get = mock_get
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "elspeth.web.auth.oidc.httpx.AsyncClient",
                return_value=client_mock,
            ),
            pytest.raises(AuthenticationError, match="JWKS unavailable"),
        ):
            await provider.authenticate("some-token")

    @pytest.mark.asyncio
    async def test_jwks_cache_expiry_refetches(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """With TTL=0, every authenticate call must re-fetch JWKS."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
        )
        token = make_rs256_token(private_key, _valid_claims())
        with mock_httpx_discovery:
            identity1 = await provider.authenticate(token)
            identity2 = await provider.authenticate(token)
        assert identity1.user_id == "user-123"
        assert identity2.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_jwks_stale_cache_served_on_fetch_failure(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """After a successful fetch, if re-fetch fails, stale cache is served."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
        )
        token = make_rs256_token(private_key, _valid_claims())

        # First call: successful fetch
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)
            assert identity.user_id == "user-123"

        # Second call: fetch fails, but stale cache should be served
        async def failing_get(url, **kwargs):
            raise httpx.ConnectError("IdP is down")

        failing_client = AsyncMock()
        failing_client.get = failing_get
        failing_client.__aenter__ = AsyncMock(return_value=failing_client)
        failing_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "elspeth.web.auth.oidc.httpx.AsyncClient",
            return_value=failing_client,
        ):
            # Should succeed using stale cache, not raise
            identity2 = await provider.authenticate(token)
            assert identity2.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_jwks_http_error_from_discovery(self) -> None:
        """HTTP 500 from discovery endpoint raises AuthenticationError."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)

        async def mock_get(url, **kwargs):
            response = MagicMock()
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Server Error",
                request=MagicMock(),
                response=MagicMock(),
            )
            return response

        client_mock = AsyncMock()
        client_mock.get = mock_get
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "elspeth.web.auth.oidc.httpx.AsyncClient",
                return_value=client_mock,
            ),
            pytest.raises(AuthenticationError, match="JWKS unavailable"),
        ):
            await provider.authenticate("some-token")


class TestOIDCJWKSShapeValidation:
    """Tests that shape-malformed IdP responses surface as AuthenticationError.

    Regression coverage for bug elspeth-c98e8e7047: a JSON-valid response
    with the wrong top-level type must not escape as TypeError/AttributeError
    (which would become HTTP 500 via middleware) and must not poison the
    JWKS cache.
    """

    @staticmethod
    def _patch_responses(discovery_json: object, keys_json: object):
        async def mock_get(url, **kwargs):
            response = MagicMock()
            response.raise_for_status = lambda: None
            if ".well-known/openid-configuration" in url:
                response.json.return_value = discovery_json
            else:
                response.json.return_value = keys_json
            return response

        client_mock = AsyncMock()
        client_mock.get = mock_get
        client_mock.__aenter__ = AsyncMock(return_value=client_mock)
        client_mock.__aexit__ = AsyncMock(return_value=False)
        return patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=client_mock)

    @pytest.mark.asyncio
    async def test_discovery_json_array_raises_auth_error(self) -> None:
        """Discovery returning a JSON array must raise AuthenticationError, not TypeError."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        with (
            self._patch_responses([], {"keys": []}),
            pytest.raises(AuthenticationError, match="not a JSON object"),
        ):
            await provider.authenticate("some-token")

    @pytest.mark.asyncio
    async def test_discovery_jwks_uri_empty_string_raises(self) -> None:
        """Discovery with empty jwks_uri must raise AuthenticationError."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        with (
            self._patch_responses({"jwks_uri": "", "issuer": ISSUER}, {"keys": []}),
            pytest.raises(AuthenticationError, match="non-empty string 'jwks_uri'"),
        ):
            await provider.authenticate("some-token")

    @pytest.mark.asyncio
    async def test_discovery_jwks_uri_non_string_raises(self) -> None:
        """Discovery with non-string jwks_uri must raise AuthenticationError."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        with (
            self._patch_responses({"jwks_uri": 42, "issuer": ISSUER}, {"keys": []}),
            pytest.raises(AuthenticationError, match="non-empty string 'jwks_uri'"),
        ):
            await provider.authenticate("some-token")

    @pytest.mark.asyncio
    async def test_jwks_json_array_raises_auth_error_and_does_not_poison_cache(self) -> None:
        """JWKS returning a JSON array must raise AuthenticationError and leave cache untouched."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        with (
            self._patch_responses({"jwks_uri": f"{ISSUER}/keys"}, []),
            pytest.raises(AuthenticationError, match="not a JSON object"),
        ):
            await provider.authenticate("some-token")
        # Cache must remain empty: a malformed response must not persist.
        assert provider._validator._jwks is None

    @pytest.mark.asyncio
    async def test_jwks_missing_keys_list_raises(self) -> None:
        """JWKS document without 'keys' list must raise AuthenticationError."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        with (
            self._patch_responses({"jwks_uri": f"{ISSUER}/keys"}, {"not_keys": []}),
            pytest.raises(AuthenticationError, match="missing 'keys' list"),
        ):
            await provider.authenticate("some-token")

    @pytest.mark.asyncio
    async def test_jwks_keys_non_list_raises(self) -> None:
        """JWKS document with non-list 'keys' must raise AuthenticationError."""
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        with (
            self._patch_responses({"jwks_uri": f"{ISSUER}/keys"}, {"keys": "not-a-list"}),
            pytest.raises(AuthenticationError, match="missing 'keys' list"),
        ):
            await provider.authenticate("some-token")

    @pytest.mark.asyncio
    async def test_jwks_inner_key_attribute_error_raises_and_does_not_cache(
        self,
        rsa_keypair,
    ) -> None:
        """Malformed inner key entries must be rejected before the cache is updated."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims())

        with (
            self._patch_responses({"jwks_uri": f"{ISSUER}/keys"}, {"keys": [42]}),
            pytest.raises(AuthenticationError, match="unusable key entries"),
        ):
            await provider.authenticate(token)

        assert provider._validator._jwks is None

    @pytest.mark.asyncio
    async def test_jwks_empty_inner_key_raises_and_does_not_cache(
        self,
        rsa_keypair,
    ) -> None:
        """Object-shaped but unusable JWKs must not poison the cache."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rs256_token(private_key, _valid_claims())

        with (
            self._patch_responses({"jwks_uri": f"{ISSUER}/keys"}, {"keys": [{}]}),
            pytest.raises(AuthenticationError, match="unusable key entries"),
        ):
            await provider.authenticate(token)

        assert provider._validator._jwks is None


class TestOIDCAlgorithmSelection:
    """Tests for algorithm selection when the JWK omits `alg`."""

    @pytest.mark.asyncio
    async def test_alg_less_rsa_jwks_accepts_ps256_token(
        self,
        rsa_keypair,
    ) -> None:
        """The token header algorithm, not PyJWT's RSA fallback, should drive verification."""
        private_key, public_key = rsa_keypair
        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        token = make_rsa_token(private_key, _valid_claims(), algorithm="PS256")

        with TestOIDCJWKSShapeValidation._patch_responses(
            {"jwks_uri": f"{ISSUER}/keys"},
            build_rsa_jwk(public_key, alg=None),
        ):
            identity = await provider.authenticate(token)

        assert identity.user_id == "user-123"
        assert identity.username == "alice"


class TestOIDCStaleCacheBackoff:
    """Tests that stale-cache fallback throttles IdP re-fetches during an outage.

    Regression coverage for bug elspeth-7f262cf7e1: once the JWKS cache
    expires during an IdP outage, stale-cache fallback must advance the
    refresh horizon so concurrent auth requests do not all queue behind
    the refresh lock re-hitting a dead IdP.
    """

    @pytest.mark.asyncio
    async def test_stale_cache_throttles_refetch_within_retry_window(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """After a failed refresh with stale cache served, subsequent calls within the retry window must not re-hit the IdP."""
        private_key, _ = rsa_keypair
        # TTL=0: every call is "past due" under the old implementation.
        # Failure retry 60s: second call within this window should NOT re-fetch.
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
            jwks_failure_retry_seconds=60,
        )
        token = make_rs256_token(private_key, _valid_claims())

        # Seed the cache with a successful fetch.
        with mock_httpx_discovery:
            await provider.authenticate(token)

        # Now simulate IdP outage. Count network attempts.
        attempt_count = 0

        async def failing_get(url, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise httpx.ConnectError("IdP is down")

        failing_client = AsyncMock()
        failing_client.get = failing_get
        failing_client.__aenter__ = AsyncMock(return_value=failing_client)
        failing_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=failing_client):
            # First call under outage: attempts one fetch, falls back to stale cache.
            identity1 = await provider.authenticate(token)
            # Second call within the 60s backoff window: MUST use cache, not re-fetch.
            identity2 = await provider.authenticate(token)
            # Third call too — still within window.
            identity3 = await provider.authenticate(token)

        assert identity1.user_id == "user-123"
        assert identity2.user_id == "user-123"
        assert identity3.user_id == "user-123"
        # The fix: only one network attempt during the backoff window.
        # Pre-fix behaviour: 3 attempts (one per authenticate call).
        assert attempt_count == 1, (
            f"Expected 1 IdP fetch during backoff window, got {attempt_count}. Stale-cache fallback is not advancing _next_refresh_at."
        )

    @pytest.mark.asyncio
    async def test_retry_window_elapses_allows_new_fetch(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Once the failure-retry window elapses, a fresh fetch is attempted."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
            jwks_failure_retry_seconds=60,
        )
        token = make_rs256_token(private_key, _valid_claims())

        # Seed the cache with a successful fetch.
        with mock_httpx_discovery:
            await provider.authenticate(token)

        attempt_count = 0

        async def failing_get(url, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise httpx.ConnectError("IdP is down")

        failing_client = AsyncMock()
        failing_client.get = failing_get
        failing_client.__aenter__ = AsyncMock(return_value=failing_client)
        failing_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=failing_client):
            await provider.authenticate(token)  # first failed fetch
            # Simulate time passing past the retry window.
            provider._validator._next_refresh_at = time.time() - 1
            await provider.authenticate(token)  # should attempt fetch again

        assert attempt_count == 2


class TestOIDCShapeFailureBackoff:
    """Shape-validation failures throttle IdP re-fetches within the retry window.

    Paired with ``TestOIDCStaleCacheBackoff`` — the network-failure
    branch already throttled; the shape-failure branch did not.
    During an outage where the IdP returns malformed JSON (proxy
    injecting an HTML error page, mid-rotation schema change, etc.),
    every concurrent ``authenticate()`` call re-entered the critical
    section and re-hit the IdP. This class pins the symmetric throttle
    contract: the current caller still sees ``AuthenticationError``
    (shape failures are not silently fallen back to stale cache), but
    subsequent callers within the retry window short-circuit at the
    top-of-function cache gate and receive the previously-validated
    cached keys. See the block comment at the top of ``ensure_jwks``
    (web/auth/oidc.py) for the full asymmetry rationale.
    """

    @pytest.mark.asyncio
    async def test_shape_failure_throttles_refetch_within_retry_window(
        self,
        rsa_keypair,
        mock_httpx_discovery,
        jwks_response,
    ) -> None:
        """Shape-malformed IdP response must advance _next_refresh_at.

        Before: every concurrent auth request in the critical section
        hit the IdP (partial DoS).  After: only the first call per
        retry window does — subsequent callers hit cache.
        """
        private_key, _ = rsa_keypair
        # TTL=0 so every call is past-due under the old implementation;
        # failure-retry=60 so the second call falls well inside the
        # throttle window.
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
            jwks_failure_retry_seconds=60,
        )
        token = make_rs256_token(private_key, _valid_claims())

        # Seed the cache with a successful fetch so the later
        # top-of-function gate has something to short-circuit on.
        with mock_httpx_discovery:
            await provider.authenticate(token)

        # Now simulate the IdP returning malformed JSON. Count network
        # attempts — the throttle bug was that each concurrent call
        # re-entered the critical section.
        attempt_count = 0

        async def shape_failing_get(url, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            response = MagicMock()
            response.raise_for_status = lambda: None
            if ".well-known/openid-configuration" in url:
                # Structurally wrong: JSON array where dict required.
                response.json.return_value = []
            else:
                response.json.return_value = {"keys": []}
            return response

        shape_client = AsyncMock()
        shape_client.get = shape_failing_get
        shape_client.__aenter__ = AsyncMock(return_value=shape_client)
        shape_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=shape_client):
            # First call: enters critical section, hits shape validator,
            # propagates AuthenticationError. This is the "do not serve
            # stale cache on the failure path" contract.
            with pytest.raises(AuthenticationError, match="not a JSON object"):
                await provider.authenticate(token)
            # Subsequent calls within the 60s window MUST NOT re-hit the
            # IdP — they short-circuit at the top-of-function cache gate
            # because _next_refresh_at was advanced by the shape-failure
            # branch.  They receive the previously-validated cached keys.
            identity2 = await provider.authenticate(token)
            identity3 = await provider.authenticate(token)

        assert identity2.user_id == "user-123"
        assert identity3.user_id == "user-123"
        # The fix: only the first call reached the IdP.  The shape
        # validator rejects the discovery document before a JWKS GET
        # is issued, so that first critical-section entry is a single
        # GET.  Subsequent calls short-circuit at the top-of-function
        # cache gate and add zero.
        #
        # Pre-fix behaviour: 3 GETs (one discovery per authenticate
        # call), because the shape-failure path re-raised without
        # advancing ``_next_refresh_at`` — every concurrent call
        # re-entered the critical section.
        assert attempt_count == 1, (
            f"Expected 1 IdP GET (the single discovery request in the single "
            f"entered critical section), got {attempt_count}. Shape-failure path "
            "is not advancing _next_refresh_at — concurrent auth requests during "
            "a shape-failure outage are re-hitting the IdP (partial-DoS regression)."
        )


class TestOIDCConcurrentStaleDuringOutage:
    """Regression coverage for bug elspeth-32982f17cf: stale-serve must
    not serialize behind the refresh lock during an IdP outage.

    Without the fast-path decoupling, concurrent auth requests during an
    IdP outage queue on ``self._jwks_lock`` while the lone fetcher blocks
    on ``httpx.get`` (up to ~15s worst case). p99 auth latency becomes
    ~15s — a partial DoS. The fix returns stale cache immediately when
    the lock is already held, so only one coroutine pays the network
    cost per retry window and followers short-circuit.
    """

    @pytest.mark.asyncio
    async def test_followers_return_stale_without_waiting_on_refresh_lock(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """During an in-flight refresh with stale cache available,
        concurrent ``authenticate()`` calls must complete without
        waiting on ``self._jwks_lock``.
        """
        private_key, _ = rsa_keypair
        # TTL=0 so every call is past-due; retry window is long so this
        # test pins lock-decoupling behaviour rather than horizon timing.
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
            jwks_failure_retry_seconds=300,
        )
        token = make_rs256_token(private_key, _valid_claims())

        # Seed the cache with a successful fetch.
        with mock_httpx_discovery:
            await provider.authenticate(token)

        # Simulate a dead IdP: first httpx.get awaits a gate we control.
        # The fix must ensure followers never await this gate.
        release = asyncio.Event()
        attempt_count = 0

        async def hanging_get(url, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            await release.wait()
            raise httpx.ConnectError("IdP is down")

        hanging_client = AsyncMock()
        hanging_client.get = hanging_get
        hanging_client.__aenter__ = AsyncMock(return_value=hanging_client)
        hanging_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=hanging_client):
            # Winner: acquires the lock and blocks inside hanging_get.
            winner = asyncio.create_task(provider.authenticate(token))
            # Poll until the winner has reached hanging_get; asyncio.Lock
            # acquisition can take several scheduler turns through the
            # httpx async context manager before reaching the await.
            for _ in range(100):
                await asyncio.sleep(0.01)
                if attempt_count == 1:
                    break
            assert attempt_count == 1, "winner never reached the hanging httpx.get"

            # Followers arrive during the winner's blocked refresh.
            follower_count = 5
            followers = [asyncio.create_task(provider.authenticate(token)) for _ in range(follower_count)]

            # Give followers a chance to run. With the lock-decoupling
            # fix they return stale immediately. Without it they queue
            # on self._jwks_lock behind the still-blocked winner.
            done, pending = await asyncio.wait(followers, timeout=1.0)
            try:
                assert not winner.done(), "winner should still be blocked in hanging_get"
                assert len(done) == follower_count, (
                    f"Expected all {follower_count} followers to short-circuit with "
                    f"stale cache, but {len(pending)} remain blocked on the refresh lock. "
                    f"Stale-serve is gated by _jwks_lock during an IdP outage — partial DoS."
                )
                # Followers did not hit the network.
                assert attempt_count == 1, f"Followers triggered additional IdP fetches (attempt_count={attempt_count})"
                # Followers all got the stale-but-valid identity.
                for task in followers:
                    assert task.result().user_id == "user-123"
            finally:
                # Unblock the winner so the test teardown is clean.
                release.set()
                await winner
                # Drain any still-pending followers (shouldn't be any with the fix).
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

    def test_default_failure_retry_seconds_is_300(self) -> None:
        """Pin the default so a silent regression is caught.

        JWKS key rotation is measured in hours/days; a 5-minute stale
        window is safe and bounds the per-retry DoS amplifier. Lower
        values re-introduce the partial DoS documented in
        elspeth-32982f17cf.
        """
        from elspeth.web.auth.oidc import JWKSTokenValidator

        for cls in (JWKSTokenValidator, OIDCAuthProvider):
            sig = inspect.signature(cls.__init__)
            assert sig.parameters["jwks_failure_retry_seconds"].default == 300, (
                f"{cls.__name__}.jwks_failure_retry_seconds default regressed below 300s — see elspeth-32982f17cf"
            )


class TestOIDCStaleCacheDoesNotLaunderProgrammerBugs:
    """Bugs in the JWKS fetch block must NOT be absorbed by the stale-cache
    fallback.

    After the shape validators (_validate_discovery_document,
    _validate_jwks_document) were added, the discovery/JWKS happy path
    cannot produce TypeError/AttributeError/KeyError from the response
    payloads — those shapes are enforced at the Tier 3 boundary as
    AuthenticationError. Any TypeError/AttributeError/KeyError that
    escapes the happy path is therefore a programmer bug in code we
    control, and must surface (per CLAUDE.md's offensive-programming
    doctrine) rather than silently produce a confident-but-wrong auth
    decision against stale keys.
    """

    @pytest.mark.asyncio
    async def test_attribute_error_propagates_does_not_serve_stale(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """An AttributeError raised inside the fetch block must propagate,
        NOT be laundered into a stale-cache fallback.

        This pins the narrowed catch in JWKSTokenValidator.ensure_jwks:
        a future regression that re-widens the catch (e.g., "catch
        everything so auth never breaks") would serve stale keys while
        masking the underlying programmer bug.
        """
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
        )
        token = make_rs256_token(private_key, _valid_claims())

        # Seed the cache with a successful fetch so there IS a stale
        # payload available — the test proves the fallback is not taken.
        with mock_httpx_discovery:
            await provider.authenticate(token)

        async def buggy_get(url, **kwargs):
            raise AttributeError("'NoneType' object has no attribute 'json'")

        buggy_client = AsyncMock()
        buggy_client.get = buggy_get
        buggy_client.__aenter__ = AsyncMock(return_value=buggy_client)
        buggy_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "elspeth.web.auth.oidc.httpx.AsyncClient",
                return_value=buggy_client,
            ),
            pytest.raises(AttributeError, match="NoneType"),
        ):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_type_error_propagates_does_not_serve_stale(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """A TypeError raised inside the fetch block must propagate."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
        )
        token = make_rs256_token(private_key, _valid_claims())

        with mock_httpx_discovery:
            await provider.authenticate(token)

        async def buggy_get(url, **kwargs):
            raise TypeError("unsupported operand type(s) for +: 'NoneType' and 'float'")

        buggy_client = AsyncMock()
        buggy_client.get = buggy_get
        buggy_client.__aenter__ = AsyncMock(return_value=buggy_client)
        buggy_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "elspeth.web.auth.oidc.httpx.AsyncClient",
                return_value=buggy_client,
            ),
            pytest.raises(TypeError, match="unsupported operand"),
        ):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_key_error_propagates_does_not_serve_stale(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """A KeyError raised inside the fetch block must propagate.

        Post-shape-validation, no KeyError can arise from IdP payload
        access (the validators use .get() and isinstance() only). Any
        KeyError is a bug in code we control.
        """
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
        )
        token = make_rs256_token(private_key, _valid_claims())

        with mock_httpx_discovery:
            await provider.authenticate(token)

        async def buggy_get(url, **kwargs):
            raise KeyError("internal_dict_lookup_bug")

        buggy_client = AsyncMock()
        buggy_client.get = buggy_get
        buggy_client.__aenter__ = AsyncMock(return_value=buggy_client)
        buggy_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "elspeth.web.auth.oidc.httpx.AsyncClient",
                return_value=buggy_client,
            ),
            pytest.raises(KeyError, match="internal_dict_lookup_bug"),
        ):
            await provider.authenticate(token)

    @pytest.mark.asyncio
    async def test_httpx_error_still_serves_stale(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """Post-narrowing, IdP outage (httpx.HTTPError) still falls back
        to stale cache. Pins that the narrowing did not accidentally
        remove the legitimate Tier 3 handling."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
        )
        token = make_rs256_token(private_key, _valid_claims())

        with mock_httpx_discovery:
            await provider.authenticate(token)

        async def outage_get(url, **kwargs):
            raise httpx.ConnectError("IdP is down")

        outage_client = AsyncMock()
        outage_client.get = outage_get
        outage_client.__aenter__ = AsyncMock(return_value=outage_client)
        outage_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "elspeth.web.auth.oidc.httpx.AsyncClient",
            return_value=outage_client,
        ):
            identity = await provider.authenticate(token)
            assert identity.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_malformed_json_still_serves_stale(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """json.JSONDecodeError (a ValueError subclass) from response.json()
        is a Tier 3 boundary failure — IdP returned non-JSON bytes — and
        must still fall through to stale cache, not crash."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
        )
        token = make_rs256_token(private_key, _valid_claims())

        with mock_httpx_discovery:
            await provider.authenticate(token)

        async def malformed_get(url, **kwargs):
            response = MagicMock()
            response.raise_for_status = lambda: None
            response.json.side_effect = ValueError("No JSON object could be decoded")
            return response

        malformed_client = AsyncMock()
        malformed_client.get = malformed_get
        malformed_client.__aenter__ = AsyncMock(return_value=malformed_client)
        malformed_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "elspeth.web.auth.oidc.httpx.AsyncClient",
            return_value=malformed_client,
        ):
            identity = await provider.authenticate(token)
            assert identity.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_invalid_url_from_idp_still_serves_stale(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """httpx.InvalidURL (NOT a subclass of httpx.HTTPError) is a
        Tier 3 boundary failure — the IdP returned a structurally-valid
        string for jwks_uri that cannot be parsed as a URL. Stale cache
        must still be served; the narrowed catch must explicitly name
        InvalidURL because it sits outside the HTTPError hierarchy.
        """
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=0,
        )
        token = make_rs256_token(private_key, _valid_claims())

        with mock_httpx_discovery:
            await provider.authenticate(token)

        async def invalid_url_get(url, **kwargs):
            raise httpx.InvalidURL("not a valid URL")

        invalid_url_client = AsyncMock()
        invalid_url_client.get = invalid_url_get
        invalid_url_client.__aenter__ = AsyncMock(return_value=invalid_url_client)
        invalid_url_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "elspeth.web.auth.oidc.httpx.AsyncClient",
            return_value=invalid_url_client,
        ):
            identity = await provider.authenticate(token)
            assert identity.user_id == "user-123"


class TestOIDCColdStartBackoff:
    """Cold-start IdP outage must not serialize every request on the
    refresh lock.

    Symmetric with ``TestOIDCStaleCacheBackoff``, but for the *empty
    cache* case: neither the top-of-function short-circuit nor the
    lock-locked short-circuit fires on cold start (both gate on
    ``self._jwks is not None``), so pre-fix every request hit the
    refresh lock and paid a full httpx timeout when the IdP was down.
    The fix advances ``_next_refresh_at`` unconditionally on fetch
    failure and adds a cold-start fail-fast that raises
    ``AuthenticationError`` while the retry window is open without
    touching the network.
    """

    @pytest.mark.asyncio
    async def test_cold_start_fetch_failure_throttles_subsequent_calls(
        self,
        rsa_keypair,
    ) -> None:
        """Cold start + IdP outage: first request fetches (and fails),
        subsequent requests within the retry window fail fast without
        re-hitting the IdP."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=3600,
            jwks_failure_retry_seconds=60,
        )
        token = make_rs256_token(private_key, _valid_claims())

        # Cache is empty (no seed fetch). Simulate IdP outage for every call.
        attempt_count = 0

        async def failing_get(url, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise httpx.ConnectError("IdP is down")

        failing_client = AsyncMock()
        failing_client.get = failing_get
        failing_client.__aenter__ = AsyncMock(return_value=failing_client)
        failing_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=failing_client):
            # First call: attempts fetch, has no stale cache, raises AuthenticationError.
            with pytest.raises(AuthenticationError):
                await provider.authenticate(token)
            # Second and third calls within the 60s backoff window:
            # MUST fail fast WITHOUT re-entering the httpx client.
            with pytest.raises(AuthenticationError):
                await provider.authenticate(token)
            with pytest.raises(AuthenticationError):
                await provider.authenticate(token)

        # The fix: only one network attempt during the backoff window.
        # Pre-fix behaviour: 3 attempts (one per authenticate call).
        assert attempt_count == 1, (
            f"Expected 1 IdP fetch during cold-start backoff window, got "
            f"{attempt_count}. Cold-start throttle is not advancing "
            f"_next_refresh_at when stale_jwks is None."
        )

    @pytest.mark.asyncio
    async def test_cold_start_retry_window_elapses_allows_new_fetch(
        self,
        rsa_keypair,
        mock_httpx_discovery,
    ) -> None:
        """After the cold-start retry window elapses, a fresh fetch is
        attempted and — with a healthy IdP — succeeds."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=3600,
            jwks_failure_retry_seconds=60,
        )
        token = make_rs256_token(private_key, _valid_claims())

        attempt_count = 0

        async def failing_get(url, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise httpx.ConnectError("IdP is down")

        failing_client = AsyncMock()
        failing_client.get = failing_get
        failing_client.__aenter__ = AsyncMock(return_value=failing_client)
        failing_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=failing_client),
            pytest.raises(AuthenticationError),
        ):
            await provider.authenticate(token)

        # Simulate time passing past the retry window.
        provider._validator._next_refresh_at = time.time() - 1

        # Now the IdP is healthy again.
        with mock_httpx_discovery:
            identity = await provider.authenticate(token)

        assert identity.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_cold_start_concurrent_requests_share_single_fetch(
        self,
        rsa_keypair,
    ) -> None:
        """Concurrent cold-start requests during an IdP outage: exactly
        one fetch attempt; all requests raise AuthenticationError; none
        block on the httpx timeout in sequence."""
        private_key, _ = rsa_keypair
        provider = OIDCAuthProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_cache_ttl_seconds=3600,
            jwks_failure_retry_seconds=60,
        )
        token = make_rs256_token(private_key, _valid_claims())

        attempt_count = 0

        async def failing_get(url, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            # Tiny sleep so queued coroutines have a chance to pile up on
            # the lock before this first attempt completes.
            await asyncio.sleep(0.01)
            raise httpx.ConnectError("IdP is down")

        failing_client = AsyncMock()
        failing_client.get = failing_get
        failing_client.__aenter__ = AsyncMock(return_value=failing_client)
        failing_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elspeth.web.auth.oidc.httpx.AsyncClient", return_value=failing_client):
            results = await asyncio.gather(
                *(provider.authenticate(token) for _ in range(5)),
                return_exceptions=True,
            )

        # All five requests failed.
        assert all(isinstance(r, AuthenticationError) for r in results), results
        # Exactly one attempt — the first acquired the lock and ran, the
        # other four hit the cold-start throttle fast-path and never
        # touched the network.
        assert attempt_count == 1, (
            f"Expected 1 IdP fetch for 5 concurrent cold-start requests, "
            f"got {attempt_count}. Cold-start throttle is not firing "
            f"inside the lock re-check."
        )


class TestOIDCProtocolConformance:
    """Verify OIDCAuthProvider satisfies the AuthProvider protocol."""

    def test_oidc_satisfies_auth_provider(self) -> None:
        from elspeth.web.auth.protocol import AuthProvider

        provider = OIDCAuthProvider(issuer=ISSUER, audience=AUDIENCE)
        assert isinstance(provider, AuthProvider)
