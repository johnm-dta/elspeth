"""OIDC authentication provider -- JWKS discovery and JWT validation.

Validates tokens issued by any OIDC-compliant identity provider.
The frontend handles the IdP redirect; this backend only validates
the resulting token.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, NoReturn, cast

import httpx
import jwt
import structlog
from jwt.exceptions import PyJWTError

from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.web.auth.models import AuthenticationError, AuthProviderUnavailable, UserIdentity, UserProfile
from elspeth.web.auth.urls import validate_oidc_issuer
from elspeth.web.validation import has_visible_content

slog = structlog.get_logger()


@trust_boundary(
    tier=3,
    source="verified Cognito access-token claims",
    source_param="payload",
    suppresses=("R1", "R5"),
    invariant="raises AuthenticationError unless client_id is the exact configured string and token_use is exactly access; never falls back to aud",
    test_ref=("tests/unit/web/auth/test_oidc_provider.py::TestOIDCAudienceClaimModes::test_manual_claim_helper_is_a_trust_boundary"),
    test_fingerprint="7a4f5811c00d10440587fefe4b61b13b1f5c1cee09940a953eee1b05a467651b",
)
def _validate_cognito_access_claims(payload: dict[str, Any], *, audience: str) -> None:
    """Bind a verified Cognito access token to its exact public app client."""
    client_id = payload["client_id"] if "client_id" in payload else None
    token_use = payload["token_use"] if "token_use" in payload else None
    if not isinstance(client_id, str) or not client_id or client_id != audience:
        raise AuthenticationError("Invalid token: client_id claim check failed")
    if not isinstance(token_use, str) or token_use != "access":
        raise AuthenticationError("Invalid token: token_use claim check failed")


def optional_profile_claim(payload: dict[str, Any], claim_name: str) -> str | None:
    """Return optional cosmetic IdP claims as visible strings or None."""
    # Tier-3 token claims: an optional claim may simply be absent. Read it
    # explicitly (membership-then-subscript) so the "absent -> None" step is
    # visible decision-making rather than a `.get()` that hides it.
    value = payload[claim_name] if claim_name in payload else None
    if value is None or not isinstance(value, str):
        return None
    claim_value = cast(str, value)
    if not has_visible_content(claim_value):
        return None
    return claim_value


class JWKSTokenValidator:
    """JWKS discovery, caching, and JWT decode -- shared by OIDC and Entra."""

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_cache_ttl_seconds: int = 3600,
        jwks_failure_retry_seconds: int = 300,
        jwks_max_stale_seconds: int = 86_400,
        *,
        audience_claim: Literal["aud", "client_id"] = "aud",
    ) -> None:
        self._issuer = validate_oidc_issuer(issuer)
        self._audience = audience
        if audience_claim not in ("aud", "client_id"):
            raise ValueError("audience_claim must be aud or client_id")
        self._audience_claim = audience_claim
        self._jwks_cache_ttl_seconds = jwks_cache_ttl_seconds
        # 300s default (5 min): JWKS keys rotate on the order of hours
        # to days, so serving stale keys for up to 5 minutes is safer
        # than forcing concurrent auth requests through a blocked
        # httpx.get to a dead IdP. Lower values amplify the per-retry
        # partial DoS described in elspeth-32982f17cf.
        self._jwks_failure_retry_seconds = jwks_failure_retry_seconds
        # Absolute upper bound on cached-key authority. Failure retries move
        # ``_next_refresh_at`` but must never renew this lifetime; only a
        # fully validated successful fetch resets ``_jwks_last_success_at``.
        self._jwks_max_stale_seconds = jwks_max_stale_seconds
        self._jwks: dict[str, Any] | None = None
        self._jwks_last_success_at: float | None = None
        self._jwks_refresh_failed = False
        # Separate "when should we try to refresh next" from "when did we
        # last succeed." A successful fetch sets this to now+ttl; a failure
        # that serves stale cache sets this to now+failure_retry so concurrent
        # auth requests during an IdP outage don't all queue behind the lock
        # re-hitting a dead IdP.
        self._next_refresh_at: float = 0.0
        self._jwks_lock = asyncio.Lock()

    def _cached_jwks_within_max_stale_age(self, now: float) -> bool:
        """Return whether cached keys still have authority at ``now``."""
        if self._jwks is None or self._jwks_last_success_at is None:
            return False
        age = now - self._jwks_last_success_at
        return 0 <= age < self._jwks_max_stale_seconds

    @staticmethod
    def _raise_max_stale_age_exceeded() -> NoReturn:
        """Fail closed without exposing IdP payloads or cache timestamps."""
        raise AuthProviderUnavailable("JWKS unavailable (cached keys exceeded maximum stale age)")

    @trust_boundary(
        tier=3,
        source="OIDC discovery document JSON fetched from the IdP's .well-known/openid-configuration endpoint",
        source_param="discovery",
        suppresses=("R1",),
        invariant="raises AuthenticationError on non-dict or missing/blank 'jwks_uri'; never coerces a malformed document",
        test_ref="tests/unit/web/auth/test_oidc_provider.py::TestJWKSValidatorBoundaryRaises::test_validate_discovery_document_non_dict_raises",
        test_fingerprint="c05f0c70cad8dd916cab0485cd40d1dea32dcedb6b6eb53b2e049e7749ea8987",
    )
    def _validate_discovery_document(self, discovery: Any) -> str:
        """Shape-validate the OIDC discovery document and return jwks_uri.

        Tier 3 boundary: an IdP (or a misbehaving proxy in front of one)
        can return JSON-valid payloads with the wrong top-level shape.
        Reject them at the boundary as ``AuthenticationError`` rather
        than letting ``TypeError``/``KeyError`` escape as HTTP 500.
        """
        if not isinstance(discovery, dict):
            raise AuthenticationError(f"OIDC discovery document is not a JSON object (got {type(discovery).__name__})")
        discovery_issuer = discovery.get("issuer")
        if not isinstance(discovery_issuer, str) or discovery_issuer != self._issuer:
            raise AuthenticationError("OIDC discovery document failed exact issuer check")
        jwks_uri = discovery.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri.strip():
            raise AuthenticationError("OIDC discovery document missing non-empty string 'jwks_uri'")
        return self._validate_jwks_uri_policy(jwks_uri)

    def _validate_jwks_uri_policy(self, jwks_uri: str) -> str:
        """Validate discovery-provided JWKS URL before fetching it."""
        try:
            issuer_url = httpx.URL(self._issuer)
            jwks_url = httpx.URL(jwks_uri)
        except httpx.InvalidURL as exc:
            raise AuthenticationError("OIDC discovery document 'jwks_uri' must be a valid URL") from exc

        if jwks_url.scheme != "https":
            raise AuthenticationError("OIDC discovery document 'jwks_uri' must be an HTTPS URL")
        if jwks_url.userinfo:
            raise AuthenticationError("OIDC discovery document 'jwks_uri' must not include embedded credentials")

        issuer_origin = (issuer_url.scheme, issuer_url.host, issuer_url.port)
        jwks_origin = (jwks_url.scheme, jwks_url.host, jwks_url.port)
        if jwks_origin != issuer_origin:
            raise AuthenticationError("OIDC discovery document 'jwks_uri' must use the same origin as issuer")

        return jwks_uri

    @staticmethod
    @trust_boundary(
        tier=3,
        source="JWKS document JSON fetched from the IdP's jwks_uri endpoint",
        source_param="jwks",
        suppresses=("R1",),
        invariant="raises AuthenticationError on non-dict or missing 'keys' list; never coerces a malformed document",
        test_ref="tests/unit/web/auth/test_oidc_provider.py::TestJWKSValidatorBoundaryRaises::test_validate_jwks_document_missing_keys_raises",
        test_fingerprint="c06b1f0b8c04a6b33dd5e1b3bec1da3752bc6081c170ae076aa175f20893e09d",
    )
    def _validate_jwks_document(jwks: Any) -> dict[str, Any]:
        """Shape-validate the JWKS document.

        Returns the same dict on success; raises ``AuthenticationError``
        on shape mismatch. Called BEFORE caching so a malformed response
        cannot poison ``self._jwks`` for the TTL window.
        """
        if not isinstance(jwks, dict):
            raise AuthenticationError(f"JWKS document is not a JSON object (got {type(jwks).__name__})")
        keys = jwks.get("keys")
        if not isinstance(keys, list):
            raise AuthenticationError("JWKS document missing 'keys' list")
        return jwks

    @staticmethod
    def _parse_jwk_set(jwks: dict[str, Any]) -> jwt.PyJWKSet:
        """Fully validate JWK usability before decode/caching decisions."""
        try:
            return jwt.PyJWKSet.from_dict(jwks)
        except (PyJWTError, AttributeError, TypeError, ValueError) as exc:
            raise AuthenticationError(f"JWKS document contains unusable key entries: {type(exc).__name__}") from exc

    @staticmethod
    @trust_boundary(
        tier=3,
        source="Unverified JWT header decoded from the externally-supplied bearer token",
        source_param="header",
        suppresses=("R1",),
        invariant="raises AuthenticationError on missing/blank/non-string 'alg'; never coerces a malformed header",
        test_ref="tests/unit/web/auth/test_oidc_provider.py::TestJWKSValidatorBoundaryRaises::test_get_token_algorithm_missing_alg_raises",
        test_fingerprint="d51616a82b5ccd166afb3b2db23de04fe312d78a61cc30b3b08a11737d9ce901",
    )
    def _get_token_algorithm(header: dict[str, Any]) -> str:
        """Return the token header algorithm as a validated non-empty string."""
        alg = header.get("alg")
        if not isinstance(alg, str) or not alg.strip():
            raise AuthenticationError("Token header missing non-empty string 'alg'")
        return alg

    @staticmethod
    @trust_boundary(
        tier=3,
        source="JWKS document JSON fetched from the IdP's jwks_uri endpoint (the matched key's 'alg' field)",
        source_param="jwks",
        suppresses=("R1",),
        invariant="raises AuthenticationError when a matched JWK advertises a non-string/blank 'alg'; returns None for honest absence (no match, or matched key omits 'alg')",
        test_ref="tests/unit/web/auth/test_oidc_provider.py::TestJWKSValidatorBoundaryRaises::test_get_jwk_algorithm_invalid_alg_raises",
        test_fingerprint="ef13d7cd4093eca4035f4cfffb768c2c7c820cad79a5c26f531b08ef47f4b4d1",
    )
    def _get_jwk_algorithm(jwks: dict[str, Any], *, kid: str | None) -> str | None:
        """Return the matched JWK's advertised algorithm, if it has one."""
        for raw_key in jwks["keys"]:
            if not isinstance(raw_key, dict):
                continue
            if raw_key.get("kid") != kid:
                continue
            alg = raw_key.get("alg")
            if alg is None:
                return None
            if not isinstance(alg, str) or not alg.strip():
                raise AuthenticationError("JWKS key has invalid non-empty string 'alg' value")
            return alg
        return None

    async def ensure_jwks(self) -> dict[str, Any]:
        """Fetch and cache JWKS keys from the OIDC discovery endpoint.

        Uses double-checked locking to prevent thundering herd at TTL
        boundary. On fetch failure, serves stale cache only within
        ``jwks_max_stale_seconds`` of the last successful fetch and advances
        the refresh horizon by ``jwks_failure_retry_seconds``
        so concurrent auth requests during an IdP outage don't all queue
        behind the lock re-hitting a dead IdP. (JWKS keys are long-lived;
        stale keys during a transient IdP blip are safer than a hard
        auth outage.)

        Followers short-circuit when a refresh is already in flight:
        if stale cache is populated and the refresh lock is held, return
        stale immediately rather than queue behind a blocked ``httpx.get``.
        Only the single lock-holder pays the network cost per retry
        window — see elspeth-32982f17cf for the partial-DoS this
        prevents.

        **Cold-start throttle:** with no cache, the stale-serve bypasses
        cannot fire (they all gate on ``self._jwks is not None``). If the
        IdP is down at cold start, every concurrent auth request would
        otherwise serialize on the refresh lock and hit the httpx timeout
        in turn. The cold-start throttle — advancing ``_next_refresh_at``
        unconditionally on fetch failure and short-circuiting requests
        while ``self._jwks is None and now < self._next_refresh_at`` —
        means only the first request per retry window pays the network
        cost, and the rest fail fast with 503 until the horizon passes.
        """
        now = time.monotonic()
        if self._jwks is not None and now < self._next_refresh_at:
            if self._cached_jwks_within_max_stale_age(now):
                return self._jwks
            # A failure retry window remains load-bearing even after cached
            # keys lose authority: fail closed until the retry horizon rather
            # than re-hitting a dead IdP on every request. If this is merely a
            # cache TTL longer than the configured hard age, fall through and
            # refresh now.
            if self._jwks_refresh_failed:
                self._raise_max_stale_age_exceeded()

        # Cold-start throttle fast-path: a prior fetch failed within the
        # current retry window AND we have no cache to serve. Fail fast
        # BEFORE touching the lock so cold-start traffic during an IdP
        # outage is shed without queueing. The ``_next_refresh_at``
        # timestamp is the single source of truth for "are we in a
        # throttle window" — see the failure branches below for where
        # it is advanced on both network and shape failures.
        if self._jwks is None and now < self._next_refresh_at:
            raise AuthProviderUnavailable("JWKS unavailable (cold-start fetch failed, retry throttled)")

        # Lock-decoupled stale-serve: if another coroutine is already
        # attempting a refresh and we have a cached (possibly stale) JWKS,
        # return it without waiting on the lock. This prevents concurrent
        # auth requests from serializing behind a dead IdP fetch (up to
        # the httpx 15s timeout worst case). The ``locked()`` check is
        # best-effort: if the lock is released between the check and our
        # acquire call, we fall through to the normal double-checked
        # locking path and the re-check inside the lock is authoritative.
        if self._jwks is not None and self._jwks_lock.locked():
            if self._cached_jwks_within_max_stale_age(now):
                return self._jwks
            self._raise_max_stale_age_exceeded()

        async with self._jwks_lock:
            # Re-check inside lock (another coroutine may have refreshed)
            now = time.monotonic()
            if self._jwks is not None and now < self._next_refresh_at:
                if self._cached_jwks_within_max_stale_age(now):
                    return self._jwks
                if self._jwks_refresh_failed:
                    self._raise_max_stale_age_exceeded()

            # Cold-start throttle inside lock: another coroutine's fetch
            # may have failed while we were queued on the lock. Repeat
            # the fail-fast check here so lock-queued cold-start requests
            # don't re-hit the dead IdP when the first coroutine releases
            # the lock after raising.
            if self._jwks is None and now < self._next_refresh_at:
                raise AuthProviderUnavailable("JWKS unavailable (cold-start fetch failed, retry throttled)")

            stale_jwks = self._jwks
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                    discovery_url = f"{self._issuer}/.well-known/openid-configuration"
                    discovery_resp = await client.get(discovery_url)
                    discovery_resp.raise_for_status()
                    jwks_uri = self._validate_discovery_document(discovery_resp.json())

                    jwks_resp = await client.get(jwks_uri)
                    jwks_resp.raise_for_status()
                    # Shape-validate BEFORE assigning to cache: a wrong-shaped
                    # response must not poison self._jwks.
                    validated = self._validate_jwks_document(jwks_resp.json())
                    self._parse_jwk_set(validated)
                    success_at = time.monotonic()
                    self._jwks = validated
                    self._jwks_last_success_at = success_at
                    self._jwks_refresh_failed = False
                    self._next_refresh_at = success_at + self._jwks_cache_ttl_seconds
            except AuthenticationError:
                # Shape-validation failure — advance the refresh horizon
                # by ``_jwks_failure_retry_seconds`` (the same throttle the
                # network-failure branch below applies) BEFORE re-raising.
                #
                # Why throttle here too: without the horizon advance, a
                # malformed-JSON outage at the IdP causes every concurrent
                # auth request in the critical section to re-hit the IdP —
                # the partial-DoS vector elspeth-32982f17cf closed for
                # network errors.  Shape-failure is functionally
                # indistinguishable at this layer (reachable IdP, bad
                # payload); the thundering herd is identical.
                #
                # Why we still re-raise (no stale-cache return on this
                # path): the CURRENT caller — who triggered the validator
                # — gets a clean 401 so an unrecoverable misconfiguration
                # (IdP rotated its document schema, corrupt reverse proxy,
                # etc.) surfaces as an auth failure rather than a silent
                # fallback.  Subsequent callers within the throttle window
                # short-circuit at the top of ``ensure_jwks`` via the
                # ``self._jwks is not None and now < self._next_refresh_at``
                # gate and receive the previously-validated cached keys —
                # symmetric with the network-failure branch's stale-serve
                # semantics, where only the first caller per window pays
                # the cost of discovering the outage.  If cache is empty
                # (shape failure during bootstrap), the top-of-function
                # gate does not trigger and the window only throttles the
                # single-caller path; every caller still gets 401 until
                # the IdP returns valid JSON.
                # Advance the horizon UNCONDITIONALLY (both warm and
                # cold-start paths). Without this, a cold-start shape
                # failure leaves ``_next_refresh_at`` at 0 and every
                # queued coroutine re-hits the malformed IdP in
                # succession. With the cold-start throttle fast-paths
                # above, setting the horizon here lets all subsequent
                # callers in the retry window fail fast at the top of
                # ``ensure_jwks`` with a clean 401.
                failure_at = time.monotonic()
                self._jwks_refresh_failed = True
                self._next_refresh_at = failure_at + self._jwks_failure_retry_seconds
                slog.debug(
                    "JWKS shape validation failed; throttling refresh",
                    issuer=self._issuer,
                    has_stale_cache=stale_jwks is not None,
                    next_refresh_in_seconds=self._jwks_failure_retry_seconds,
                )
                if stale_jwks is not None and not self._cached_jwks_within_max_stale_age(failure_at):
                    self._raise_max_stale_age_exceeded()
                raise
            except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
                # Narrowed from the historical (HTTPError, KeyError, ValueError,
                # TypeError, AttributeError) catch so that programmer-bug
                # exceptions no longer launder into a stale-cache fallback.
                #
                # After the shape validators (_validate_discovery_document,
                # _validate_jwks_document) were added, IdP payload access at
                # this Tier 3 boundary cannot produce KeyError / TypeError /
                # AttributeError on the happy path — those shapes are
                # rejected upstream as AuthenticationError. Anything in
                # those classes reaching this catch would therefore be a
                # bug in the surrounding try block, and suppressing it to
                # serve stale keys would produce a confident-but-wrong
                # auth decision (CLAUDE.md's "silent wrong result is worse
                # than a crash" rule).
                #
                # The remaining catches preserve the legitimate Tier 3
                # failure modes that must serve stale cache:
                #   - httpx.HTTPError: connect/read timeouts, HTTP 5xx from
                #     the IdP, transport errors. Base class of
                #     RequestError / TransportError / ConnectError /
                #     TimeoutException / HTTPStatusError (raised by
                #     response.raise_for_status()).
                #   - httpx.InvalidURL: explicitly named because it sits
                #     OUTSIDE the HTTPError hierarchy (direct Exception
                #     subclass). Fires when jwks_uri is a non-empty string
                #     but not a parseable URL — the shape validator only
                #     checks the string-ness, not URL syntax, so the IdP
                #     can still feed us junk here.
                #   - ValueError: covers json.JSONDecodeError and
                #     UnicodeDecodeError from response.json() when the
                #     IdP returns non-JSON or mis-encoded bytes.
                # Advance the horizon UNCONDITIONALLY (both stale-serve
                # and cold-start paths). The original code only advanced
                # when ``stale_jwks is not None`` — cold-start outages
                # therefore left ``_next_refresh_at`` at 0 and every
                # concurrent auth request serialized on ``self._jwks_lock``
                # through a full httpx timeout apiece, which is the
                # documented-but-live DoS vector the cold-start throttle
                # (above) was added to close. Writing the horizon here is
                # the same source-of-truth update that makes the
                # fast-paths at the top of ``ensure_jwks`` fire.
                failure_at = time.monotonic()
                self._jwks_refresh_failed = True
                self._next_refresh_at = failure_at + self._jwks_failure_retry_seconds
                if stale_jwks is not None and self._cached_jwks_within_max_stale_age(failure_at):
                    # Serve stale cache -- JWKS keys are long-lived
                    slog.debug(
                        "JWKS fetch failed, serving stale cache",
                        issuer=self._issuer,
                        exc_class=type(exc).__name__,
                        next_refresh_in_seconds=self._jwks_failure_retry_seconds,
                    )
                    return stale_jwks
                if stale_jwks is not None:
                    slog.debug(
                        "JWKS fetch failed after cached keys exceeded maximum stale age",
                        issuer=self._issuer,
                        exc_class=type(exc).__name__,
                        max_stale_seconds=self._jwks_max_stale_seconds,
                        next_refresh_in_seconds=self._jwks_failure_retry_seconds,
                    )
                    self._raise_max_stale_age_exceeded()
                slog.debug(
                    "JWKS cold-start fetch failed; throttling retry",
                    issuer=self._issuer,
                    exc_class=type(exc).__name__,
                    next_refresh_in_seconds=self._jwks_failure_retry_seconds,
                )
                # Class name only. ``str(exc)`` on httpx.InvalidURL carries
                # the raw jwks_uri (Tier-3 IdP-provided string), and
                # httpx.ConnectError can include the resolved IP of the IdP.
                # ``AuthProviderUnavailable.detail`` flows verbatim into the 503
                # response body via auth middleware, so payload-free text is
                # the only safe channel here. Symmetric with the Tier-1
                # redaction discipline applied to _handle_plugin_crash
                # (routes.py) and the blob/plugin SQLAlchemyError sites.
                raise AuthProviderUnavailable(f"JWKS unavailable: {type(exc).__name__}") from exc

        return self._jwks

    def decode_token(self, token: str, jwks: dict[str, Any]) -> dict[str, Any]:
        """Decode and validate a JWT using the cached JWKS.

        Extracts the signing key from the JWKS by matching the token's
        ``kid`` header to the correct JWK entry.
        """
        try:
            header = jwt.get_unverified_header(token)
            token_alg = self._get_token_algorithm(header)
            kid = header.get("kid")
            jwk_set = self._parse_jwk_set(jwks)
            matched_jwk = None
            for key in jwk_set.keys:
                if key.key_id == kid:
                    matched_jwk = key
                    break
            if matched_jwk is None:
                raise AuthenticationError("Invalid token: signing key check failed")
            jwk_alg = self._get_jwk_algorithm(jwks, kid=kid)
            if jwk_alg is not None and jwk_alg != token_alg:
                raise AuthenticationError("Invalid token: algorithm check failed")
            if self._audience_claim == "client_id":
                if token_alg != "RS256":
                    raise AuthenticationError("Invalid token: Cognito algorithm check failed")
                payload = jwt.decode(
                    token,
                    matched_jwk.key,
                    algorithms=["RS256"],
                    issuer=self._issuer,
                    options={
                        "verify_aud": False,
                        "require": ["exp", "iat", "iss", "sub", "client_id", "token_use"],
                    },
                )
                _validate_cognito_access_claims(payload, audience=self._audience)
            else:
                payload = jwt.decode(
                    token,
                    matched_jwk.key,
                    algorithms=[token_alg],
                    audience=self._audience,
                    issuer=self._issuer,
                    options={"require": ["exp"]},
                )
        except PyJWTError as exc:
            # Class name only. PyJWT exception messages may echo claim
            # values (e.g. "Audience doesn't match. Expected: ... Got: ...")
            # or token segments in decode errors, which AuthenticationError
            # would surface into the 401 response body.
            raise AuthenticationError(f"Invalid token: {type(exc).__name__}") from exc
        return payload


class OIDCAuthProvider:
    """Validates OIDC tokens via JWKS discovery."""

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_cache_ttl_seconds: int = 3600,
        jwks_failure_retry_seconds: int = 300,
        jwks_max_stale_seconds: int = 86_400,
        *,
        audience_claim: Literal["aud", "client_id"] = "aud",
    ) -> None:
        self._validator = JWKSTokenValidator(
            issuer,
            audience,
            jwks_cache_ttl_seconds,
            jwks_failure_retry_seconds,
            jwks_max_stale_seconds,
            audience_claim=audience_claim,
        )

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate an OIDC token and return the authenticated identity."""
        jwks = await self._validator.ensure_jwks()
        payload = self._validator.decode_token(token, jwks)

        try:
            sub = payload["sub"]
        except KeyError as exc:
            raise AuthenticationError("Missing required 'sub' claim in token") from exc

        # preferred_username is an optional cosmetic claim. Decide the username
        # explicitly: use the IdP-supplied visible value when present, otherwise
        # fall back to the canonical `sub` identifier (always a valid principal).
        preferred_username = self._optional_profile_claim(payload, "preferred_username")
        username = preferred_username if preferred_username is not None else sub

        return UserIdentity(
            user_id=sub,
            username=username,
        )

    @staticmethod
    def _optional_profile_claim(payload: dict[str, Any], claim_name: str) -> str | None:
        """Return optional cosmetic claims as visible strings or None."""
        return optional_profile_claim(payload, claim_name)

    @trust_boundary(
        tier=3,
        source="OIDC bearer access token from a remote IdP; decoded payload carries optional profile claims including 'groups'",
        source_param="token",
        suppresses=("R1",),
        invariant="raises AuthenticationError on malformed non-list 'groups'; treats absent 'groups' as no groups; never coerces scalar groups silently",
        test_ref="tests/unit/web/auth/test_oidc_provider.py::TestOIDCGetUserInfo::test_non_list_groups_claim_raises",
        test_fingerprint="cefa7844868a4e9b7662d3966a910dc1698332f477a06c5b9050ddb699657898",
    )
    async def get_user_info(self, token: str) -> UserProfile:
        """Decode the OIDC token and extract profile claims."""
        jwks = await self._validator.ensure_jwks()
        payload = self._validator.decode_token(token, jwks)

        try:
            sub = payload["sub"]
        except KeyError as exc:
            raise AuthenticationError("Missing required 'sub' claim in token") from exc

        raw_groups = payload.get("groups")
        if raw_groups is None:
            groups: list[str] = []
        elif isinstance(raw_groups, list):
            # Coerce group IDs to str — IdPs may send integers (e.g. Entra
            # group object IDs). This is intentional Tier 3 coercion.
            groups = [str(g) for g in raw_groups]
        else:
            raise AuthenticationError(
                f"Unexpected type for 'groups' claim: {type(raw_groups).__name__} (expected list) — check IdP token configuration"
            )

        display_name = self._optional_profile_claim(payload, "name")
        if display_name is None:
            display_name = self._optional_profile_claim(payload, "preferred_username")

        # preferred_username is an optional cosmetic claim. Decide the username
        # explicitly: use the IdP-supplied visible value when present, otherwise
        # fall back to the canonical `sub` identifier (always a valid principal).
        preferred_username = self._optional_profile_claim(payload, "preferred_username")
        username = preferred_username if preferred_username is not None else sub

        return UserProfile(
            user_id=sub,
            username=username,
            display_name=display_name,
            email=self._optional_profile_claim(payload, "email"),
            groups=tuple(groups),
        )
