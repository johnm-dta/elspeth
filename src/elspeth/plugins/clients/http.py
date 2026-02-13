# src/elspeth/plugins/clients/http.py
"""Audited HTTP client with automatic call recording.

Provides SSRF-safe HTTP methods via get_ssrf_safe() which uses IP pinning
to prevent DNS rebinding attacks. See core/security/web.py for details.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from datetime import UTC, datetime
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.core.canonical import stable_hash
from elspeth.core.security.web import (
    SSRFSafeRequest,
    validate_url_for_ssrf,
)
from elspeth.plugins.clients.base import AuditedClientBase, TelemetryEmitCallback

logger = structlog.get_logger(__name__)


def _contains_non_finite(obj: Any) -> bool:
    """Recursively check if object contains NaN or Infinity float values.

    This is a Tier 3 boundary check: external JSON may contain non-finite values
    (Python's json module accepts them), but canonicalization rejects them. We
    detect these at the HTTP boundary to record as parse failure rather than
    crashing during audit recording.

    Args:
        obj: Any JSON-parsed value (dict, list, or primitive)

    Returns:
        True if any float value is NaN or Infinity
    """
    if isinstance(obj, float):
        return math.isnan(obj) or math.isinf(obj)
    if isinstance(obj, dict):
        return any(_contains_non_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_non_finite(v) for v in obj)
    return False


def _parse_json_strict(text: str) -> tuple[Any, str | None]:
    """Parse JSON with strict rejection of NaN/Infinity.

    Python's stdlib json module accepts non-finite values by default, but
    these cannot be canonicalized. This function parses and validates in
    one step at the Tier 3 boundary.

    Args:
        text: JSON string to parse

    Returns:
        Tuple of (parsed_value, error_message)
        - On success: (parsed_dict_or_list, None)
        - On failure: (None, error_message)
    """
    try:
        parsed = json.loads(text)
    except JSONDecodeError as e:
        return None, str(e)

    # Check for non-finite values that canonicalization would reject
    if _contains_non_finite(parsed):
        return None, "JSON contains non-finite values (NaN or Infinity)"

    return parsed, None


if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class AuditedHTTPClient(AuditedClientBase):
    """HTTP client that automatically records all calls to audit trail.

    Wraps httpx to ensure every HTTP call is recorded to the Landscape
    audit trail. Supports:
    - Automatic request/response recording
    - Auth header fingerprinting (HMAC fingerprint stored, not raw secrets)
    - Latency measurement
    - Error recording
    - Telemetry emission after successful audit recording
    - Rate limiting (when limiter provided)

    Example:
        client = AuditedHTTPClient(
            recorder=recorder,
            state_id=state_id,
            run_id=run_id,
            telemetry_emit=telemetry_emit,
            base_url="https://api.example.com",
            headers={"Authorization": "Bearer ..."},
            limiter=registry.get_limiter("api.example.com"),
        )

        response = client.post("/v1/process", json={"data": "value"})
        print(response.json())
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        state_id: str,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        *,
        timeout: float = 30.0,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        limiter: Any = None,  # RateLimiter | NoOpLimiter | None
        token_id: str | None = None,
    ) -> None:
        """Initialize audited HTTP client.

        Args:
            recorder: LandscapeRecorder for audit trail storage
            state_id: Node state ID to associate calls with
            run_id: Pipeline run ID for telemetry correlation
            telemetry_emit: Callback to emit telemetry events
            timeout: Request timeout in seconds (default: 30.0)
            base_url: Optional base URL to prepend to all requests
            headers: Default headers for all requests
            limiter: Optional rate limiter for throttling requests
            token_id: Optional token identity for telemetry correlation
        """
        super().__init__(recorder, state_id, run_id, telemetry_emit, limiter=limiter, token_id=token_id)
        self._timeout = timeout
        self._base_url = base_url
        self._default_headers = headers or {}
        # Shared httpx.Client for connection pooling and TCP reuse.
        # httpx.Client is thread-safe; the internal pool handles concurrency.
        # Per-request timeouts override the default via timeout= kwarg.
        # follow_redirects=False: SSRF-safe methods manage redirects manually.
        self._client = httpx.Client(
            timeout=self._timeout,
            follow_redirects=False,
        )

    # Well-known sensitive headers (exact match, case-insensitive).
    # Checked first for O(1) lookup before falling back to word matching.
    _SENSITIVE_HEADERS_EXACT = frozenset(
        {
            # Request headers
            "authorization",
            "proxy-authorization",
            "cookie",
            "x-api-key",
            "api-key",
            "x-auth-token",
            "x-access-token",
            "x-csrf-token",
            "x-xsrf-token",
            "ocp-apim-subscription-key",
            # Response headers
            "set-cookie",
            "www-authenticate",
            "proxy-authenticate",
        }
    )

    # Words that indicate sensitive content when they appear as complete
    # delimiter-separated segments in header names.
    # e.g. "X-Auth-Token" splits to {"x","auth","token"} → matches "auth","token"
    # but "X-Author" splits to {"x","author"} → no match (avoids false positives)
    _SENSITIVE_HEADER_WORDS = frozenset(
        {
            "auth",
            "authkey",
            "authtoken",
            "accesstoken",
            "apikey",
            "authorization",
            "key",
            "secret",
            "token",
            "password",
            "credential",
        }
    )

    def _is_sensitive_header(self, header_name: str) -> bool:
        """Check if a header name indicates sensitive content.

        Uses delimiter-separated word matching to avoid false positives from
        broad substring matching (e.g. "key" in "monkey", "auth" in "author"),
        while still catching common compact forms like ``apikey`` and
        ``authkey``.

        Args:
            header_name: Header name to check

        Returns:
            True if header likely contains secrets
        """
        lower_name = header_name.lower()
        if lower_name in self._SENSITIVE_HEADERS_EXACT:
            return True
        segments = [seg for seg in re.split(r"[^a-z0-9]+", lower_name) if seg]
        if any(seg in self._SENSITIVE_HEADER_WORDS for seg in segments):
            return True
        return lower_name.startswith("x") and lower_name[1:] in self._SENSITIVE_HEADER_WORDS

    def _filter_request_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Fingerprint sensitive request headers for audit recording.

        Sensitive headers (auth, api keys, tokens) are replaced with HMAC
        fingerprints so that:
        1. Raw secrets are NEVER stored in the audit trail
        2. Different credentials produce different fingerprints
        3. Replay/verify can distinguish requests by credential identity

        In dev mode (ELSPETH_ALLOW_RAW_SECRETS=true), sensitive headers are
        removed entirely (no fingerprint key required).

        Args:
            headers: Full headers dict

        Returns:
            Headers dict with sensitive values fingerprinted (or removed in dev mode)
        """
        from elspeth.core.security import get_fingerprint_key, secret_fingerprint

        # Check if fingerprint key is available
        allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"

        try:
            get_fingerprint_key()
            have_key = True
        except ValueError:
            have_key = False

        result: dict[str, str] = {}

        for k, v in headers.items():
            if self._is_sensitive_header(k):
                if allow_raw:
                    # Dev mode: remove header (don't store secrets, don't require key)
                    pass
                elif have_key:
                    # Fingerprint the sensitive value
                    fp = secret_fingerprint(v)
                    result[k] = f"<fingerprint:{fp}>"
                else:
                    # No key and not dev mode - this shouldn't happen in production
                    # Remove header to avoid leaking secrets (fail-safe)
                    logger.warning(
                        "Sensitive header '%s' dropped: no fingerprint key available. "
                        "Set ELSPETH_FINGERPRINT_KEY or ELSPETH_ALLOW_RAW_SECRETS=true",
                        k,
                    )
                    # Don't include this header
            else:
                # Non-sensitive header: include as-is
                result[k] = v

        return result

    def _filter_response_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Filter out sensitive response headers from audit recording.

        Response headers that may contain secrets (cookies, auth challenges)
        are not recorded. Uses the same word-boundary matching as request
        header filtering.

        Args:
            headers: Full headers dict

        Returns:
            Headers dict with sensitive headers removed
        """
        return {k: v for k, v in headers.items() if not self._is_sensitive_header(k)}

    def _extract_provider(self, url: str) -> str:
        """Extract provider (host) from URL for telemetry.

        SECURITY: This method MUST NOT leak credentials. URLs may contain
        embedded userinfo (e.g., https://user:pass@host/). We use hostname
        only, which strips the userinfo component per RFC 3986.

        Args:
            url: Full URL

        Returns:
            Host portion of URL (e.g., "api.example.com"), without credentials
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        # SECURITY: Use hostname (not netloc) to avoid leaking credentials.
        # netloc = [userinfo@]host[:port], hostname = just the host
        return parsed.hostname or "unknown"

    def close(self) -> None:
        """Close the underlying httpx client and release connections."""
        self._client.close()

    def _resolve_url(self, url: str) -> str:
        """Join base_url with path, handling slash combinations."""
        if self._base_url:
            base = self._base_url.rstrip("/")
            path = url.lstrip("/")
            return f"{base}/{path}"
        return url

    def _parse_response_body(self, response: httpx.Response, full_url: str) -> Any:
        """Parse response body at Tier 3 boundary with strict JSON validation.

        Handles JSON (with NaN/Infinity rejection), text, and binary content.
        """
        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            parsed, error = _parse_json_strict(response.text)
            if error is not None:
                logger.warning(
                    "JSON parse failed despite Content-Type: application/json",
                    extra={
                        "url": full_url,
                        "status_code": response.status_code,
                        "body_preview": response.text[:200],
                        "error": error,
                    },
                )
                return {
                    "_json_parse_failed": True,
                    "_error": error,
                    "_raw_text": response.text[:10_000],
                }
            return parsed

        # Text content types: text/*, application/xml, application/x-www-form-urlencoded
        is_text_content = content_type.startswith("text/") or "xml" in content_type or "form-urlencoded" in content_type
        if is_text_content:
            return response.text

        # Binary content as base64 for JSON serialization
        return {"_binary": base64.b64encode(response.content).decode("ascii")}

    def _record_and_emit(
        self,
        *,
        call_index: int,
        full_url: str,
        request_data: dict[str, Any],
        response: httpx.Response | None,
        response_data: dict[str, Any] | None,
        error_data: dict[str, Any] | None,
        latency_ms: float,
        call_status: CallStatus,
        token_id_override: str | None = None,
    ) -> None:
        """Record call to audit trail and emit telemetry event.

        Args:
            token_id_override: Per-call token_id for telemetry. When provided,
                overrides the client-level token_id. Used by batch transforms
                where a single client serves multiple tokens.
        """
        self._recorder.record_call(
            state_id=self._state_id,
            call_index=call_index,
            call_type=CallType.HTTP,
            status=call_status,
            request_data=request_data,
            response_data=response_data,
            error=error_data,
            latency_ms=latency_ms,
        )

        # Telemetry emitted AFTER successful Landscape recording
        # Wrapped in try/except to prevent telemetry failures from corrupting audit trail
        effective_token_id = token_id_override if token_id_override is not None else self._telemetry_token_id()
        try:
            self._telemetry_emit(
                ExternalCallCompleted(
                    timestamp=datetime.now(UTC),
                    run_id=self._run_id,
                    call_type=CallType.HTTP,
                    provider=self._extract_provider(full_url),
                    status=call_status,
                    latency_ms=latency_ms,
                    state_id=self._state_id,
                    operation_id=None,
                    token_id=effective_token_id,
                    request_hash=stable_hash(request_data),
                    response_hash=stable_hash(response_data) if response_data else None,
                    request_payload=request_data,
                    response_payload=response_data,
                    token_usage=None,
                )
            )
        except Exception as tel_err:
            logger.warning(
                "telemetry_emit_failed",
                error=str(tel_err),
                error_type=type(tel_err).__name__,
                run_id=self._run_id,
                state_id=self._state_id,
                call_type="http",
            )

    def _execute_request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        timeout: float | None,
        json: dict[str, Any] | None = None,
        params: dict[str, str | int | float] | None = None,
        token_id: str | None = None,
    ) -> httpx.Response:
        """Execute an HTTP request with audit recording and telemetry.

        Shared implementation for post() and get(). Handles URL resolution,
        header merging, response parsing, audit recording, and telemetry.

        Args:
            method: HTTP method ("POST" or "GET")
            url: URL path (appended to base_url if configured)
            headers: Additional headers for this request
            timeout: Request timeout override (uses client default if None)
            json: JSON body (POST only)
            params: Query parameters (GET only)
            token_id: Per-call token_id for telemetry (overrides client default).
                Used by batch transforms where one client serves multiple tokens.

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: For network/HTTP errors
        """
        self._acquire_rate_limit()
        call_index = self._next_call_index()

        full_url = self._resolve_url(url)
        merged_headers = {**self._default_headers, **(headers or {})}
        effective_timeout = timeout if timeout is not None else self._timeout

        # Build request data for audit trail (method-specific fields always present
        # for stable schema per method)
        request_data: dict[str, Any] = {
            "method": method,
            "url": full_url,
            "headers": self._filter_request_headers(merged_headers),
        }
        if method == "POST":
            request_data["json"] = json
        elif method == "GET":
            request_data["params"] = params

        start = time.perf_counter()

        try:
            # Dispatch to the correct httpx method
            if method == "POST":
                response = self._client.post(
                    full_url,
                    json=json,
                    headers=merged_headers,
                    timeout=effective_timeout,
                )
            else:
                response = self._client.get(
                    full_url,
                    params=params,
                    headers=merged_headers,
                    timeout=effective_timeout,
                )

            latency_ms = (time.perf_counter() - start) * 1000

            response_body = self._parse_response_body(response, full_url)

            # 2xx = SUCCESS, 4xx/5xx = ERROR
            is_success = 200 <= response.status_code < 300
            call_status = CallStatus.SUCCESS if is_success else CallStatus.ERROR

            response_data: dict[str, Any] = {
                "status_code": response.status_code,
                "headers": self._filter_response_headers(dict(response.headers)),
                "body_size": len(response.content),
                "body": response_body,
            }

            error_data: dict[str, Any] | None = None
            if not is_success:
                error_data = {
                    "type": "HTTPError",
                    "message": f"HTTP {response.status_code}",
                    "status_code": response.status_code,
                }

            self._record_and_emit(
                call_index=call_index,
                full_url=full_url,
                request_data=request_data,
                response=response,
                response_data=response_data,
                error_data=error_data,
                latency_ms=latency_ms,
                call_status=call_status,
                token_id_override=token_id,
            )

            return response

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            self._record_and_emit(
                call_index=call_index,
                full_url=full_url,
                request_data=request_data,
                response=None,
                response_data=None,
                error_data={
                    "type": type(e).__name__,
                    "message": str(e),
                },
                latency_ms=latency_ms,
                call_status=CallStatus.ERROR,
                token_id_override=token_id,
            )

            raise

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        token_id: str | None = None,
    ) -> httpx.Response:
        """Make POST request with automatic audit recording.

        Args:
            url: URL path (appended to base_url if configured)
            json: JSON body to send (optional)
            headers: Additional headers for this request
            timeout: Request timeout in seconds (uses client default if None)
            token_id: Per-call token_id for telemetry (overrides client default).
                Used by batch transforms where one client serves multiple tokens.

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: For network/HTTP errors
        """
        return self._execute_request(
            method="POST",
            url=url,
            headers=headers,
            timeout=timeout,
            json=json,
            token_id=token_id,
        )

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str | int | float] | None = None,
        timeout: float | None = None,
        token_id: str | None = None,
    ) -> httpx.Response:
        """Make GET request with automatic audit recording.

        Args:
            url: URL path (appended to base_url if configured)
            headers: Additional headers for this request
            params: Query parameters to append to URL
            timeout: Request timeout in seconds (uses client default if None)
            token_id: Per-call token_id for telemetry (overrides client default).
                Used by batch transforms where one client serves multiple tokens.

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: For network/HTTP errors
        """
        return self._execute_request(
            method="GET",
            url=url,
            headers=headers,
            timeout=timeout,
            params=params,
            token_id=token_id,
        )

    def get_ssrf_safe(
        self,
        request: SSRFSafeRequest,
        *,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = False,
        max_redirects: int = 10,
    ) -> httpx.Response:
        """GET with SSRF-safe IP pinning and redirect validation.

        Connects to the pre-validated IP in the SSRFSafeRequest, setting the
        Host header and TLS SNI to the original hostname. Each redirect hop
        is independently validated against the SSRF blocklist.

        Args:
            request: SSRFSafeRequest from validate_url_for_ssrf()
            headers: Additional headers for this request
            follow_redirects: Whether to follow HTTP redirects (default: False)
            max_redirects: Maximum redirect hops when follow_redirects=True

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: For network/HTTP errors
            SSRFBlockedError: If redirect target resolves to blocked IP
        """
        self._acquire_rate_limit()

        call_index = self._next_call_index()

        merged_headers = {
            **self._default_headers,
            **(headers or {}),
            "Host": request.host_header,
        }

        connection_url = request.connection_url
        effective_timeout = self._timeout

        # TLS SNI: use original hostname for certificate verification
        extensions: dict[str, str] = {}
        if request.scheme == "https":
            extensions["sni_hostname"] = request.sni_hostname

        # Record original URL and resolved IP in audit trail
        request_data: dict[str, Any] = {
            "method": "GET",
            "url": request.original_url,
            "resolved_ip": request.resolved_ip,
            "headers": self._filter_request_headers(merged_headers),
        }

        start = time.perf_counter()

        try:
            # Ephemeral client for SSRF-safe requests: connection_url uses the
            # resolved IP (e.g. https://1.2.3.4:443/path), so all hostnames sharing
            # an IP would map to the same pool key. A shared client would reuse a
            # TLS connection established for hostname-A when requesting hostname-B,
            # silently skipping SNI negotiation and certificate verification.
            with httpx.Client(
                timeout=effective_timeout,
                follow_redirects=False,
            ) as ssrf_client:
                response = ssrf_client.get(
                    connection_url,
                    headers=merged_headers,
                    extensions=extensions if extensions else None,
                )

            # Handle redirects with SSRF validation at each hop
            redirect_count = 0
            if follow_redirects:
                response, redirect_count = self._follow_redirects_safe(
                    response,
                    max_redirects,
                    effective_timeout,
                    merged_headers,
                    original_url=request.original_url,
                )

            latency_ms = (time.perf_counter() - start) * 1000

            response_body: Any = None
            content_type = response.headers.get("content-type", "")

            if "application/json" in content_type:
                parsed, error = _parse_json_strict(response.text)
                if error is not None:
                    logger.warning(
                        "JSON parse failed despite Content-Type: application/json",
                        extra={
                            "url": request.original_url,
                            "status_code": response.status_code,
                            "body_preview": response.text[:200],
                            "error": error,
                        },
                    )
                    response_body = {
                        "_json_parse_failed": True,
                        "_error": error,
                        "_raw_text": response.text[:10_000],
                    }
                else:
                    response_body = parsed
            else:
                is_text_content = content_type.startswith("text/") or "xml" in content_type or "form-urlencoded" in content_type
                if is_text_content:
                    response_body = response.text
                else:
                    response_body = {"_binary": base64.b64encode(response.content).decode("ascii")}

            is_success = 200 <= response.status_code < 300
            call_status = CallStatus.SUCCESS if is_success else CallStatus.ERROR

            response_data: dict[str, Any] = {
                "status_code": response.status_code,
                "headers": self._filter_response_headers(dict(response.headers)),
                "body_size": len(response.content),
                "body": response_body,
            }
            if redirect_count > 0:
                response_data["redirect_count"] = redirect_count

            error_data: dict[str, Any] | None = None
            if not is_success:
                error_data = {
                    "type": "HTTPError",
                    "message": f"HTTP {response.status_code}",
                    "status_code": response.status_code,
                }

            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.HTTP,
                status=call_status,
                request_data=request_data,
                response_data=response_data,
                error=error_data,
                latency_ms=latency_ms,
            )

            try:
                self._telemetry_emit(
                    ExternalCallCompleted(
                        timestamp=datetime.now(UTC),
                        run_id=self._run_id,
                        call_type=CallType.HTTP,
                        provider=request.host_header,
                        status=call_status,
                        latency_ms=latency_ms,
                        state_id=self._state_id,
                        operation_id=None,
                        token_id=self._telemetry_token_id(),
                        request_hash=stable_hash(request_data),
                        response_hash=stable_hash(response_data),
                        request_payload=request_data,
                        response_payload=response_data,
                        token_usage=None,
                    )
                )
            except Exception as tel_err:
                logger.warning(
                    "telemetry_emit_failed",
                    error=str(tel_err),
                    error_type=type(tel_err).__name__,
                    run_id=self._run_id,
                    state_id=self._state_id,
                    call_type="http_ssrf_safe",
                )

            return response

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=request_data,
                error={
                    "type": type(e).__name__,
                    "message": str(e),
                },
                latency_ms=latency_ms,
            )

            try:
                self._telemetry_emit(
                    ExternalCallCompleted(
                        timestamp=datetime.now(UTC),
                        run_id=self._run_id,
                        call_type=CallType.HTTP,
                        provider=request.host_header,
                        status=CallStatus.ERROR,
                        latency_ms=latency_ms,
                        state_id=self._state_id,
                        operation_id=None,
                        token_id=self._telemetry_token_id(),
                        request_hash=stable_hash(request_data),
                        response_hash=None,
                        request_payload=request_data,
                        response_payload=None,
                        token_usage=None,
                    )
                )
            except Exception as tel_err:
                logger.warning(
                    "telemetry_emit_failed",
                    error=str(tel_err),
                    error_type=type(tel_err).__name__,
                    run_id=self._run_id,
                    state_id=self._state_id,
                    call_type="http_ssrf_safe",
                )

            raise

    def _follow_redirects_safe(
        self,
        response: httpx.Response,
        max_redirects: int,
        timeout: float,
        original_headers: dict[str, str],
        original_url: str,
    ) -> tuple[httpx.Response, int]:
        """Follow HTTP redirects with SSRF validation at each hop.

        Each redirect target is independently resolved and validated against
        the SSRF blocklist, preventing redirect-based SSRF attacks like:
        attacker.com -> 301 -> http://169.254.169.254/

        Args:
            response: Initial response (may be a redirect)
            max_redirects: Maximum number of redirect hops
            timeout: Request timeout for each hop
            original_headers: Headers from original request (minus Host, which is set per-hop)
            original_url: Hostname-based URL for resolving relative redirects.
                response.url is IP-based (from connection_url rewrite), so relative
                Location headers must resolve against the original hostname URL to
                preserve correct Host headers and TLS SNI.

        Returns:
            Tuple of (final non-redirect response, number of redirects followed)

        Raises:
            SSRFBlockedError: If any redirect target resolves to a blocked IP
            httpx.TooManyRedirects: If redirect chain exceeds max_redirects
        """
        redirects_followed = 0
        # Track the logical hostname URL for resolving relative redirects.
        # response.url is IP-based (from connection_url), so relative Location
        # headers would resolve against the IP instead of the hostname.
        hostname_url = httpx.URL(original_url)

        while response.is_redirect and redirects_followed < max_redirects:
            location = response.headers.get("location")
            if not location:
                break

            # Capture the URL we're redirecting FROM (before updating hostname_url)
            redirect_from = str(hostname_url)

            # Resolve relative URLs against the hostname URL, NOT response.url
            redirect_url = str(hostname_url.join(location))

            # CRITICAL: Validate the redirect target for SSRF
            redirect_request = validate_url_for_ssrf(redirect_url)

            # Update hostname_url to the redirect target for the next iteration.
            # If this was an absolute redirect to a different host, hostname_url
            # now tracks that new host.
            hostname_url = httpx.URL(redirect_url)

            # Build headers for this hop (Host header for virtual hosting)
            hop_headers = {k: v for k, v in original_headers.items() if k.lower() != "host"}
            hop_headers["Host"] = redirect_request.host_header

            # TLS SNI for this hop
            extensions: dict[str, str] = {}
            if redirect_request.scheme == "https":
                extensions["sni_hostname"] = redirect_request.sni_hostname

            hop_start = time.perf_counter()

            # Ephemeral client per redirect hop: same TLS/SNI isolation rationale
            # as the initial SSRF-safe request — IP-based connection_url would
            # cause the pool to reuse connections across different hostnames.
            with httpx.Client(
                timeout=timeout,
                follow_redirects=False,
            ) as hop_client:
                response = hop_client.get(
                    redirect_request.connection_url,
                    headers=hop_headers,
                    extensions=extensions if extensions else None,
                )

            hop_latency_ms = (time.perf_counter() - hop_start) * 1000
            redirects_followed += 1

            # Record this redirect hop in the audit trail.
            # Each hop is a real network call — it may hit a different server.
            hop_call_index = self._next_call_index()
            hop_request_data = {
                "method": "GET",
                "url": redirect_url,
                "resolved_ip": redirect_request.resolved_ip,
                "hop_number": redirects_followed,
                "redirect_from": redirect_from,
                "headers": self._filter_request_headers(hop_headers),
            }
            hop_response_data = {
                "status_code": response.status_code,
                "headers": self._filter_response_headers(dict(response.headers)),
            }

            self._recorder.record_call(
                state_id=self._state_id,
                call_index=hop_call_index,
                call_type=CallType.HTTP_REDIRECT,
                status=CallStatus.SUCCESS if response.status_code < 400 else CallStatus.ERROR,
                request_data=hop_request_data,
                response_data=hop_response_data,
                latency_ms=hop_latency_ms,
            )

        if response.is_redirect and redirects_followed >= max_redirects:
            raise httpx.TooManyRedirects(
                f"Exceeded {max_redirects} redirects",
                request=response.request,
            )

        return response, redirects_followed
