"""OIDC authentication provider -- JWKS discovery and JWT validation.

Validates tokens issued by any OIDC-compliant identity provider.
The frontend handles the IdP redirect; this backend only validates
the resulting token.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import httpx
import jwt
import structlog
from jwt.exceptions import PyJWTError

from elspeth.web.auth.models import AuthenticationError, UserIdentity, UserProfile
from elspeth.web.validation import has_visible_content

slog = structlog.get_logger()


class JWKSTokenValidator:
    """JWKS discovery, caching, and JWT decode -- shared by OIDC and Entra."""

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_cache_ttl_seconds: int = 3600,
        jwks_failure_retry_seconds: int = 300,
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._jwks_cache_ttl_seconds = jwks_cache_ttl_seconds
        # 300s default (5 min): JWKS keys rotate on the order of hours
        # to days, so serving stale keys for up to 5 minutes is safer
        # than forcing concurrent auth requests through a blocked
        # httpx.get to a dead IdP. Lower values amplify the per-retry
        # partial DoS described in elspeth-32982f17cf.
        self._jwks_failure_retry_seconds = jwks_failure_retry_seconds
        self._jwks: dict[str, Any] | None = None
        # Separate "when should we try to refresh next" from "when did we
        # last succeed." A successful fetch sets this to now+ttl; a failure
        # that serves stale cache sets this to now+failure_retry so concurrent
        # auth requests during an IdP outage don't all queue behind the lock
        # re-hitting a dead IdP.
        self._next_refresh_at: float = 0.0
        self._jwks_lock = asyncio.Lock()

    @staticmethod
    def _validate_discovery_document(discovery: Any) -> str:
        """Shape-validate the OIDC discovery document and return jwks_uri.

        Tier 3 boundary: an IdP (or a misbehaving proxy in front of one)
        can return JSON-valid payloads with the wrong top-level shape.
        Reject them at the boundary as ``AuthenticationError`` rather
        than letting ``TypeError``/``KeyError`` escape as HTTP 500.
        """
        if not isinstance(discovery, dict):
            raise AuthenticationError(f"OIDC discovery document is not a JSON object (got {type(discovery).__name__})")
        jwks_uri = discovery.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri.strip():
            raise AuthenticationError("OIDC discovery document missing non-empty string 'jwks_uri'")
        return jwks_uri

    @staticmethod
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
    def _get_token_algorithm(header: dict[str, Any]) -> str:
        """Return the token header algorithm as a validated non-empty string."""
        alg = header.get("alg")
        if not isinstance(alg, str) or not alg.strip():
            raise AuthenticationError("Token header missing non-empty string 'alg'")
        return alg

    @staticmethod
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
                raise AuthenticationError(f"JWKS key for kid={kid!r} has invalid non-empty string 'alg' value")
            return alg
        return None

    async def ensure_jwks(self) -> dict[str, Any]:
        """Fetch and cache JWKS keys from the OIDC discovery endpoint.

        Uses double-checked locking to prevent thundering herd at TTL
        boundary. On fetch failure, serves stale cache if available
        and advances the refresh horizon by ``jwks_failure_retry_seconds``
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
        cost, and the rest fail fast with 401 until the horizon passes.
        """
        now = time.time()
        if self._jwks is not None and now < self._next_refresh_at:
            return self._jwks

        # Cold-start throttle fast-path: a prior fetch failed within the
        # current retry window AND we have no cache to serve. Fail fast
        # BEFORE touching the lock so cold-start traffic during an IdP
        # outage is shed without queueing. The ``_next_refresh_at``
        # timestamp is the single source of truth for "are we in a
        # throttle window" — see the failure branches below for where
        # it is advanced on both network and shape failures.
        if self._jwks is None and now < self._next_refresh_at:
            raise AuthenticationError("JWKS unavailable (cold-start fetch failed, retry throttled)")

        # Lock-decoupled stale-serve: if another coroutine is already
        # attempting a refresh and we have a cached (possibly stale) JWKS,
        # return it without waiting on the lock. This prevents concurrent
        # auth requests from serializing behind a dead IdP fetch (up to
        # the httpx 15s timeout worst case). The ``locked()`` check is
        # best-effort: if the lock is released between the check and our
        # acquire call, we fall through to the normal double-checked
        # locking path and the re-check inside the lock is authoritative.
        if self._jwks is not None and self._jwks_lock.locked():
            return self._jwks

        async with self._jwks_lock:
            # Re-check inside lock (another coroutine may have refreshed)
            now = time.time()
            if self._jwks is not None and now < self._next_refresh_at:
                return self._jwks

            # Cold-start throttle inside lock: another coroutine's fetch
            # may have failed while we were queued on the lock. Repeat
            # the fail-fast check here so lock-queued cold-start requests
            # don't re-hit the dead IdP when the first coroutine releases
            # the lock after raising.
            if self._jwks is None and now < self._next_refresh_at:
                raise AuthenticationError("JWKS unavailable (cold-start fetch failed, retry throttled)")

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
                    self._jwks = validated
                    self._next_refresh_at = now + self._jwks_cache_ttl_seconds
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
                self._next_refresh_at = now + self._jwks_failure_retry_seconds
                slog.debug(
                    "JWKS shape validation failed; throttling refresh",
                    issuer=self._issuer,
                    has_stale_cache=stale_jwks is not None,
                    next_refresh_in_seconds=self._jwks_failure_retry_seconds,
                )
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
                self._next_refresh_at = now + self._jwks_failure_retry_seconds
                if stale_jwks is not None:
                    # Serve stale cache -- JWKS keys are long-lived
                    slog.debug(
                        "JWKS fetch failed, serving stale cache",
                        issuer=self._issuer,
                        error=str(exc),
                        next_refresh_in_seconds=self._jwks_failure_retry_seconds,
                    )
                    return stale_jwks
                slog.debug(
                    "JWKS cold-start fetch failed; throttling retry",
                    issuer=self._issuer,
                    error=str(exc),
                    next_refresh_in_seconds=self._jwks_failure_retry_seconds,
                )
                # Class name only. ``str(exc)`` on httpx.InvalidURL carries
                # the raw jwks_uri (Tier-3 IdP-provided string), and
                # httpx.ConnectError can include the resolved IP of the IdP.
                # ``AuthenticationError.detail`` flows verbatim into the 401
                # response body via auth middleware, so payload-free text is
                # the only safe channel here. Symmetric with the Tier-1
                # redaction discipline applied to _handle_plugin_crash
                # (routes.py) and the blob/plugin SQLAlchemyError sites.
                raise AuthenticationError(f"JWKS unavailable: {type(exc).__name__}") from exc

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
                raise AuthenticationError(f"No matching key found in JWKS for kid={kid!r}")
            jwk_alg = self._get_jwk_algorithm(jwks, kid=kid)
            if jwk_alg is not None and jwk_alg != token_alg:
                raise AuthenticationError(f"Token header alg {token_alg!r} does not match JWKS alg {jwk_alg!r} for kid={kid!r}")
            payload: dict[str, Any] = jwt.decode(
                token,
                matched_jwk.key,
                algorithms=[token_alg],
                audience=self._audience,
                issuer=self._issuer,
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
    ) -> None:
        self._validator = JWKSTokenValidator(
            issuer,
            audience,
            jwks_cache_ttl_seconds,
            jwks_failure_retry_seconds,
        )

    async def authenticate(self, token: str) -> UserIdentity:
        """Validate an OIDC token and return the authenticated identity."""
        jwks = await self._validator.ensure_jwks()
        payload = self._validator.decode_token(token, jwks)

        try:
            sub = payload["sub"]
        except KeyError as exc:
            raise AuthenticationError("Missing required 'sub' claim in token") from exc

        return UserIdentity(
            user_id=sub,
            # preferred_username is optional — fall back to sub if absent,
            # null, or empty.
            username=payload.get("preferred_username") or sub,
        )

    @staticmethod
    def _optional_profile_claim(payload: dict[str, Any], claim_name: str) -> str | None:
        """Return optional cosmetic claims as visible strings or None."""
        value = payload.get(claim_name)
        if value is None or not isinstance(value, str):
            return None
        claim_value = cast(str, value)
        if not has_visible_content(claim_value):
            return None
        return claim_value

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

        return UserProfile(
            user_id=sub,
            # preferred_username is optional — fall back to sub if absent,
            # null, or empty. IdPs may send null for this optional claim.
            username=payload.get("preferred_username") or sub,
            display_name=display_name,
            email=self._optional_profile_claim(payload, "email"),
            groups=tuple(groups),
        )
