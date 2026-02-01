# src/elspeth/plugins/clients/http.py
"""Audited HTTP client with automatic call recording."""

from __future__ import annotations

import base64
import os
import time
from datetime import UTC, datetime
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from elspeth.contracts import CallStatus, CallType
from elspeth.core.canonical import stable_hash
from elspeth.plugins.clients.base import AuditedClientBase, TelemetryEmitCallback
from elspeth.telemetry.events import ExternalCallCompleted

logger = structlog.get_logger(__name__)

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
        """
        super().__init__(recorder, state_id, run_id, telemetry_emit, limiter=limiter)
        self._timeout = timeout
        self._base_url = base_url
        self._default_headers = headers or {}

    # Headers that may contain secrets - fingerprinted in audit trail
    _SENSITIVE_REQUEST_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "x-auth-token", "proxy-authorization"})
    _SENSITIVE_RESPONSE_HEADERS = frozenset({"set-cookie", "www-authenticate", "proxy-authenticate", "x-auth-token"})

    def _is_sensitive_header(self, header_name: str) -> bool:
        """Check if a header name indicates sensitive content.

        Args:
            header_name: Header name to check

        Returns:
            True if header likely contains secrets
        """
        lower_name = header_name.lower()
        return (
            lower_name in self._SENSITIVE_REQUEST_HEADERS
            or "auth" in lower_name
            or "key" in lower_name
            or "secret" in lower_name
            or "token" in lower_name
        )

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
                if have_key:
                    # Fingerprint the sensitive value
                    fp = secret_fingerprint(v)
                    result[k] = f"<fingerprint:{fp}>"
                elif not allow_raw:
                    # No key and not dev mode - this shouldn't happen in production
                    # Remove header to avoid leaking secrets (fail-safe)
                    logger.warning(
                        "Sensitive header '%s' dropped: no fingerprint key available. "
                        "Set ELSPETH_FINGERPRINT_KEY or ELSPETH_ALLOW_RAW_SECRETS=true",
                        k,
                    )
                    # Don't include this header
                else:
                    # Dev mode: remove header (don't store secrets, don't require key)
                    pass
            else:
                # Non-sensitive header: include as-is
                result[k] = v

        return result

    def _filter_response_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Filter out sensitive response headers from audit recording.

        Response headers that may contain secrets (cookies, auth challenges)
        are not recorded.

        Args:
            headers: Full headers dict

        Returns:
            Headers dict with sensitive headers removed
        """
        return {
            k: v
            for k, v in headers.items()
            if k.lower() not in self._SENSITIVE_RESPONSE_HEADERS
            and "auth" not in k.lower()
            and "key" not in k.lower()
            and "secret" not in k.lower()
            and "token" not in k.lower()
        }

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

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Make POST request with automatic audit recording.

        Args:
            url: URL path (appended to base_url if configured)
            json: JSON body to send (optional)
            headers: Additional headers for this request
            timeout: Request timeout in seconds (uses client default if None)

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: For network/HTTP errors
        """
        # Acquire rate limit permission before making external call
        self._acquire_rate_limit()

        call_index = self._next_call_index()

        # Properly join base_url and url, handling slash combinations
        if self._base_url:
            # Normalize: strip trailing slash from base, leading slash from path
            # Then join with exactly one slash
            base = self._base_url.rstrip("/")
            path = url.lstrip("/")
            full_url = f"{base}/{path}"
        else:
            full_url = url
        merged_headers = {**self._default_headers, **(headers or {})}
        effective_timeout = timeout if timeout is not None else self._timeout

        # Filter sensitive headers from recorded request
        request_data = {
            "method": "POST",
            "url": full_url,
            "json": json,
            "headers": self._filter_request_headers(merged_headers),
        }

        start = time.perf_counter()

        try:
            with httpx.Client(timeout=effective_timeout) as client:
                response = client.post(
                    full_url,
                    json=json,
                    headers=merged_headers,
                )

            latency_ms = (time.perf_counter() - start) * 1000

            # Build response data with full body for audit trail
            # Try JSON first, then text, then binary (no truncation - payload store handles size)
            response_body: Any = None
            content_type = response.headers.get("content-type", "")

            if "application/json" in content_type:
                try:
                    response_body = response.json()
                except JSONDecodeError as e:
                    # JSON parse failed despite Content-Type claiming JSON
                    # This is a Tier 3 boundary issue - external data doesn't match contract
                    # Record the failure explicitly for audit trail completeness
                    logger.warning(
                        "JSON parse failed despite Content-Type: application/json",
                        extra={
                            "url": full_url,
                            "status_code": response.status_code,
                            "body_preview": response.text[:200],
                            "error": str(e),
                        },
                    )
                    response_body = {
                        "_json_parse_failed": True,
                        "_error": str(e),
                        "_raw_text": response.text,
                    }
            else:
                # For non-JSON, detect text vs binary content
                # Text content types: text/*, application/xml, application/x-www-form-urlencoded
                is_text_content = content_type.startswith("text/") or "xml" in content_type or "form-urlencoded" in content_type

                if is_text_content:
                    # Store full text (payload store auto-persist handles large responses)
                    response_body = response.text
                else:
                    # Store binary content as base64 for JSON serialization
                    # This handles images, PDFs, and other binary formats
                    response_body = {"_binary": base64.b64encode(response.content).decode("ascii")}

            # Determine status based on HTTP response code
            # 2xx = SUCCESS, 4xx/5xx = ERROR (audit must reflect what application sees)
            is_success = 200 <= response.status_code < 300
            call_status = CallStatus.SUCCESS if is_success else CallStatus.ERROR

            response_data: dict[str, Any] = {
                "status_code": response.status_code,
                "headers": self._filter_response_headers(dict(response.headers)),
                "body_size": len(response.content),
                "body": response_body,
            }

            # For error responses, also include error details
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

            # Telemetry emitted AFTER successful Landscape recording
            # Wrapped in try/except to prevent telemetry failures from corrupting audit trail
            try:
                self._telemetry_emit(
                    ExternalCallCompleted(
                        timestamp=datetime.now(UTC),
                        run_id=self._run_id,
                        call_type=CallType.HTTP,
                        provider=self._extract_provider(full_url),
                        status=call_status,
                        latency_ms=latency_ms,
                        state_id=self._state_id,  # Transform context
                        operation_id=None,  # Not in source/sink context
                        request_hash=stable_hash(request_data),
                        response_hash=stable_hash(response_data),
                        request_payload=request_data,  # Full request for observability
                        response_payload=response_data,  # Full response for observability
                        token_usage=None,  # HTTP calls don't have token usage
                    )
                )
            except Exception as tel_err:
                # Telemetry failure must not corrupt the successful call
                logger.warning(
                    "telemetry_emit_failed",
                    error=str(tel_err),
                    error_type=type(tel_err).__name__,
                    run_id=self._run_id,
                    state_id=self._state_id,
                    call_type="http",
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

            # Telemetry emitted AFTER successful Landscape recording (even for call errors)
            # Wrapped in try/except to prevent telemetry failures from corrupting error handling
            try:
                self._telemetry_emit(
                    ExternalCallCompleted(
                        timestamp=datetime.now(UTC),
                        run_id=self._run_id,
                        call_type=CallType.HTTP,
                        provider=self._extract_provider(full_url),
                        status=CallStatus.ERROR,
                        latency_ms=latency_ms,
                        state_id=self._state_id,  # Transform context
                        operation_id=None,  # Not in source/sink context
                        request_hash=stable_hash(request_data),
                        response_hash=None,  # No response on exception
                        request_payload=request_data,  # Full request for observability
                        response_payload=None,  # No response on exception
                        token_usage=None,
                    )
                )
            except Exception as tel_err:
                # Telemetry failure must not corrupt the error handling flow
                logger.warning(
                    "telemetry_emit_failed",
                    error=str(tel_err),
                    error_type=type(tel_err).__name__,
                    run_id=self._run_id,
                    state_id=self._state_id,
                    call_type="http",
                )

            raise
