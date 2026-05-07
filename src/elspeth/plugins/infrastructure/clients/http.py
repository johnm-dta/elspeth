"""Audited HTTP client with automatic call recording.

Provides SSRF-safe HTTP methods via get_ssrf_safe() which uses IP pinning
to prevent DNS rebinding attacks. See core/security/web.py for details.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from ipaddress import IPv4Network, IPv6Network
from typing import TYPE_CHECKING, Any

import httpx
import structlog

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.call_data import CallPayload, HTTPCallError, HTTPCallRequest, HTTPCallResponse
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.core.canonical import stable_hash
from elspeth.core.security.web import (
    SSRFSafeRequest,
    validate_url_for_ssrf,
)
from elspeth.plugins.infrastructure.clients.base import AuditedClientBase, TelemetryEmitCallback
from elspeth.plugins.infrastructure.clients.fingerprinting import (
    filter_response_headers as _filter_response_headers,
)
from elspeth.plugins.infrastructure.clients.fingerprinting import (
    fingerprint_headers as _fingerprint_headers,
)
from elspeth.plugins.infrastructure.clients.fingerprinting import (
    is_sensitive_header as _is_sensitive_header_fn,
)
from elspeth.plugins.infrastructure.clients.json_utils import parse_json_strict as _parse_json_strict

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from elspeth.contracts import Call
    from elspeth.contracts.audit_protocols import CallRecorder
    from elspeth.core.rate_limit import NoOpLimiter
    from elspeth.core.rate_limit.limiter import RateLimiter


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
            execution=execution,
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
        execution: CallRecorder,
        state_id: str,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        *,
        timeout: float = 30.0,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        limiter: RateLimiter | NoOpLimiter | None = None,
        token_id: str | None = None,
    ) -> None:
        """Initialize audited HTTP client.

        Args:
            execution: CallRecorder for audit trail storage
            state_id: Node state ID to associate calls with
            run_id: Pipeline run ID for telemetry correlation
            telemetry_emit: Callback to emit telemetry events
            timeout: Request timeout in seconds (default: 30.0)
            base_url: Optional base URL to prepend to all requests
            headers: Default headers for all requests
            limiter: Optional rate limiter for throttling requests
            token_id: Optional token identity for telemetry correlation
        """
        super().__init__(execution, state_id, run_id, telemetry_emit, limiter=limiter, token_id=token_id)
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

    # Delegate to shared module functions. Instance methods preserved for
    # call-site compatibility within this class.
    def _is_sensitive_header(self, header_name: str) -> bool:
        return _is_sensitive_header_fn(header_name)

    def _filter_request_headers(self, headers: dict[str, str]) -> dict[str, str]:
        return _fingerprint_headers(headers)

    def _filter_response_headers(self, headers: dict[str, str]) -> dict[str, str]:
        return _filter_response_headers(headers)

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
                # JSON parse failure is captured in the audit trail via the
                # _json_parse_failed sentinel dict recorded by record_call().
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
        request_data: Mapping[str, Any],
        response: httpx.Response | None,
        response_data: Mapping[str, Any] | None,
        error_data: CallPayload | None,
        latency_ms: float,
        call_status: CallStatus,
        request_payload: CallPayload,
        response_payload: CallPayload | None = None,
        token_id_override: str | None = None,
    ) -> Call:
        """Record call to audit trail and emit telemetry event.

        Args:
            request_payload: Typed DTO for telemetry (e.g., HTTPCallRequest).
            response_payload: Typed DTO for telemetry (e.g., HTTPCallResponse).
            token_id_override: Per-call token_id for telemetry. When provided,
                overrides the client-level token_id. Used by batch transforms
                where a single client serves multiple tokens.

        Returns:
            Call object from Landscape recording (contains request_ref and response_ref blob hashes).
        """
        call = self._execution.record_call(
            state_id=self._state_id,
            call_index=call_index,
            call_type=CallType.HTTP,
            status=call_status,
            request_data=request_payload,
            response_data=response_payload,
            error=error_data,
            latency_ms=latency_ms,
        )

        # Telemetry emitted AFTER successful Landscape recording
        self._emit_telemetry_after_audit(
            provider=self._extract_provider(full_url),
            call_status=call_status,
            latency_ms=latency_ms,
            request_data=request_data,
            response_data=response_data,
            request_payload=request_payload,
            response_payload=response_payload,
            token_id_override=token_id_override,
            call_type_label="http",
        )

        return call

    def _emit_telemetry_after_audit(
        self,
        *,
        provider: str,
        call_status: CallStatus,
        latency_ms: float,
        request_data: Mapping[str, Any],
        response_data: Mapping[str, Any] | None,
        request_payload: CallPayload,
        response_payload: CallPayload | None = None,
        token_id_override: str | None = None,
        call_type_label: str,
    ) -> None:
        """Emit HTTP telemetry after audit recording, crashing on programmer bugs."""

        effective_token_id = token_id_override if token_id_override is not None else self._telemetry_token_id()
        try:
            self._telemetry_emit(
                ExternalCallCompleted(
                    timestamp=datetime.now(UTC),
                    run_id=self._run_id,
                    call_type=CallType.HTTP,
                    provider=provider,
                    status=call_status,
                    latency_ms=latency_ms,
                    state_id=self._state_id,
                    operation_id=None,
                    token_id=effective_token_id,
                    request_hash=stable_hash(request_data),
                    response_hash=stable_hash(response_data) if response_data else None,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    token_usage=None,
                )
            )
        except contract_errors.TIER_1_ERRORS:
            raise  # System bugs and audit integrity violations must crash
        except (TypeError, AttributeError, KeyError, NameError):
            raise  # Programming errors must crash
        except Exception as tel_err:
            logger.warning(
                "telemetry_emit_failed",
                error=str(tel_err),
                error_type=type(tel_err).__name__,
                run_id=self._run_id,
                state_id=self._state_id,
                call_type=call_type_label,
                exc_info=True,
            )

    def _build_response_payload(
        self, response: httpx.Response, full_url: str, *, redirect_count: int = 0
    ) -> tuple[HTTPCallResponse, dict[str, Any]]:
        """Build typed and dict response payloads from an HTTP response."""
        response_body = self._parse_response_body(response, full_url)
        response_dto = HTTPCallResponse(
            status_code=response.status_code,
            headers=self._filter_response_headers(dict(response.headers)),
            body_size=len(response.content),
            body=response_body,
            redirect_count=redirect_count,
        )
        return response_dto, response_dto.to_dict()

    def _execute_request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        timeout: float | None,
        json: dict[str, Any] | None = None,
        params: dict[str, str | int | float] | None = None,
        audit_request_metadata: Mapping[str, Any] | None = None,
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
            audit_request_metadata: Audit-only metadata to include in the
                recorded request payload without sending it to the remote
                service.
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

        # Build request DTO for audit trail — dataclass handles method-specific
        # field inclusion via to_dict() (POST includes json, GET includes params).
        # DTO stays alive for typed telemetry payload; dict form used for Landscape hashing.
        request_dto = HTTPCallRequest(
            method=method,
            url=full_url,
            headers=self._filter_request_headers(merged_headers),
            json=json,
            params=params,
            audit_metadata=audit_request_metadata,
        )
        request_data = request_dto.to_dict()

        start = time.perf_counter()
        response: httpx.Response | None = None

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
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            self._record_and_emit(
                call_index=call_index,
                full_url=full_url,
                request_data=request_data,
                response=None,
                response_data=None,
                error_data=HTTPCallError(
                    type=type(e).__name__,
                    message=str(e),
                ),
                latency_ms=latency_ms,
                call_status=CallStatus.ERROR,
                request_payload=request_dto,
                response_payload=None,
                token_id_override=token_id,
            )

            raise

        latency_ms = (time.perf_counter() - start) * 1000

        # 2xx = SUCCESS, 4xx/5xx = ERROR
        is_success = 200 <= response.status_code < 300
        call_status = CallStatus.SUCCESS if is_success else CallStatus.ERROR
        response_dto, response_data = self._build_response_payload(response, full_url)

        error_data: CallPayload | None = None
        if not is_success:
            error_data = HTTPCallError(
                type="HTTPError",
                message=f"HTTP {response.status_code}",
                status_code=response.status_code,
            )

        self._record_and_emit(
            call_index=call_index,
            full_url=full_url,
            request_data=request_data,
            response=response,
            response_data=response_data,
            error_data=error_data,
            latency_ms=latency_ms,
            call_status=call_status,
            request_payload=request_dto,
            response_payload=response_dto,
            token_id_override=token_id,
        )

        return response

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        audit_request_metadata: Mapping[str, Any] | None = None,
        token_id: str | None = None,
    ) -> httpx.Response:
        """Make POST request with automatic audit recording.

        Args:
            url: URL path (appended to base_url if configured)
            json: JSON body to send (optional)
            headers: Additional headers for this request
            timeout: Request timeout in seconds (uses client default if None)
            audit_request_metadata: Audit-only metadata to include in the
                recorded request payload without sending it to the remote
                service.
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
            audit_request_metadata=audit_request_metadata,
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
        allowed_ranges: Sequence[IPv4Network | IPv6Network] = (),
    ) -> tuple[httpx.Response, str, Call]:
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
            Tuple of (httpx.Response, final hostname URL as string, Call).
            The hostname URL is the logical URL after all redirects —
            distinct from response.url which is IP-based due to SSRF pinning.
            When follow_redirects is False or no redirects occurred, this is
            the original request URL.
            The Call contains request_ref and response_ref blob hashes from
            the audit trail.

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

        # Record original URL and resolved IP in audit trail.
        # DTO stays alive for typed telemetry payload; dict form used for Landscape hashing.
        request_dto = HTTPCallRequest(
            method="GET",
            url=request.original_url,
            headers=self._filter_request_headers(merged_headers),
            resolved_ip=request.resolved_ip,
        )
        request_data = request_dto.to_dict()

        start = time.perf_counter()
        response: httpx.Response | None = None

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
            final_hostname_url = request.original_url
            if follow_redirects:
                response, redirect_count, final_hostname_url = self._follow_redirects_safe(
                    response,
                    max_redirects,
                    effective_timeout,
                    merged_headers,
                    original_url=request.original_url,
                    allowed_ranges=allowed_ranges,
                )

            latency_ms = (time.perf_counter() - start) * 1000

            is_success = 200 <= response.status_code < 300
            call_status = CallStatus.SUCCESS if is_success else CallStatus.ERROR

            response_dto, response_data = self._build_response_payload(response, final_hostname_url, redirect_count=redirect_count)

            error_data: CallPayload | None = None
            if not is_success:
                error_data = HTTPCallError(
                    type="HTTPError",
                    message=f"HTTP {response.status_code}",
                    status_code=response.status_code,
                )

            call = self._execution.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.HTTP,
                status=call_status,
                request_data=request_dto,
                response_data=response_dto,
                error=error_data,
                latency_ms=latency_ms,
            )

            self._emit_telemetry_after_audit(
                provider=request.host_header,
                call_status=call_status,
                latency_ms=latency_ms,
                request_data=request_data,
                response_data=response_data,
                request_payload=request_dto,
                response_payload=response_dto,
                call_type_label="http_ssrf_safe",
            )

            return response, final_hostname_url, call

        except contract_errors.TIER_1_ERRORS:
            # Telemetry re-raise after successful Landscape record_call.
            # The SUCCESS record already exists — do NOT record a second
            # ERROR call with the same call_index (unique constraint).
            raise
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            response_payload: HTTPCallResponse | None = None
            error_response_data: Mapping[str, Any] | None = None
            if response is not None:
                response_payload, error_response_data = self._build_response_payload(response, request.original_url)

            _ = self._execution.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=request_dto,
                response_data=response_payload,
                error=HTTPCallError(
                    type=type(e).__name__,
                    message=str(e),
                ),
                latency_ms=latency_ms,
            )

            self._emit_telemetry_after_audit(
                provider=request.host_header,
                call_status=CallStatus.ERROR,
                latency_ms=latency_ms,
                request_data=request_data,
                response_data=error_response_data,
                request_payload=request_dto,
                response_payload=response_payload,
                call_type_label="http_ssrf_safe",
            )

            raise

    def _follow_redirects_safe(
        self,
        response: httpx.Response,
        max_redirects: int,
        timeout: float,
        original_headers: dict[str, str],
        original_url: str,
        *,
        allowed_ranges: Sequence[IPv4Network | IPv6Network] = (),
    ) -> tuple[httpx.Response, int, str]:
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
            Tuple of (final non-redirect response, number of redirects followed,
            final hostname URL as string). The hostname URL is the logical URL
            after all redirects — distinct from response.url which is IP-based
            due to SSRF pinning.

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

            hop_call_index = self._next_call_index()
            hop_number = redirects_followed + 1
            hop_start = time.perf_counter()

            hop_headers = {k: v for k, v in original_headers.items() if k.lower() != "host"}
            redirect_url_obj = httpx.URL(redirect_url)
            redirect_host = redirect_url_obj.host or "unknown"
            default_port = 443 if redirect_url_obj.scheme == "https" else 80
            redirect_port = redirect_url_obj.port or default_port
            hop_headers["Host"] = f"{redirect_host}:{redirect_port}" if redirect_port != default_port else redirect_host

            blocked_hop_request_dto = HTTPCallRequest(
                method="GET",
                url=redirect_url,
                headers=self._filter_request_headers(hop_headers),
                hop_number=hop_number,
                redirect_from=redirect_from,
            )

            # CRITICAL: Validate the redirect target for SSRF
            try:
                redirect_request = validate_url_for_ssrf(redirect_url, allowed_ranges=allowed_ranges)
            except contract_errors.TIER_1_ERRORS:
                raise
            except Exception as redirect_err:
                hop_latency_ms = (time.perf_counter() - hop_start) * 1000
                self._execution.record_call(
                    state_id=self._state_id,
                    call_index=hop_call_index,
                    call_type=CallType.HTTP_REDIRECT,
                    status=CallStatus.ERROR,
                    request_data=blocked_hop_request_dto,
                    error=HTTPCallError(
                        type=type(redirect_err).__name__,
                        message=str(redirect_err),
                    ),
                    latency_ms=hop_latency_ms,
                )
                raise

            # Update hostname_url to the redirect target for the next iteration.
            # If this was an absolute redirect to a different host, hostname_url
            # now tracks that new host.
            hostname_url = httpx.URL(redirect_url)

            # Build headers for this hop (Host header for virtual hosting)
            hop_headers["Host"] = redirect_request.host_header

            # TLS SNI for this hop
            extensions: dict[str, str] = {}
            if redirect_request.scheme == "https":
                extensions["sni_hostname"] = redirect_request.sni_hostname

            # Acquire rate limit for each redirect hop — each hop is a separate
            # outbound network request that must be throttled independently.
            # Bug fix: redirect hops were bypassing the rate limiter.
            self._acquire_rate_limit()

            # Pre-allocate call index and request data BEFORE the hop so that
            # both success and failure paths can record the hop in the audit trail.
            hop_request_dto = HTTPCallRequest(
                method="GET",
                url=redirect_url,
                headers=self._filter_request_headers(hop_headers),
                resolved_ip=redirect_request.resolved_ip,
                hop_number=hop_number,
                redirect_from=redirect_from,
            )
            redirects_followed += 1

            # Ephemeral client per redirect hop: same TLS/SNI isolation rationale
            # as the initial SSRF-safe request — IP-based connection_url would
            # cause the pool to reuse connections across different hostnames.
            try:
                with httpx.Client(
                    timeout=timeout,
                    follow_redirects=False,
                ) as hop_client:
                    response = hop_client.get(
                        redirect_request.connection_url,
                        headers=hop_headers,
                        extensions=extensions if extensions else None,
                    )
            except Exception as hop_err:
                hop_latency_ms = (time.perf_counter() - hop_start) * 1000
                # Record the failed hop in the audit trail so lineage is complete
                self._execution.record_call(
                    state_id=self._state_id,
                    call_index=hop_call_index,
                    call_type=CallType.HTTP_REDIRECT,
                    status=CallStatus.ERROR,
                    request_data=hop_request_dto,
                    error=HTTPCallError(
                        type=type(hop_err).__name__,
                        message=str(hop_err),
                    ),
                    latency_ms=hop_latency_ms,
                )
                raise

            hop_latency_ms = (time.perf_counter() - hop_start) * 1000

            # Record this redirect hop in the audit trail.
            # Each hop is a real network call — it may hit a different server.
            hop_response_dto = HTTPCallResponse(
                status_code=response.status_code,
                headers=self._filter_response_headers(dict(response.headers)),
            )

            self._execution.record_call(
                state_id=self._state_id,
                call_index=hop_call_index,
                call_type=CallType.HTTP_REDIRECT,
                status=CallStatus.SUCCESS if response.status_code < 400 else CallStatus.ERROR,
                request_data=hop_request_dto,
                response_data=hop_response_dto,
                latency_ms=hop_latency_ms,
            )

        if response.is_redirect and redirects_followed >= max_redirects:
            raise httpx.TooManyRedirects(
                f"Exceeded {max_redirects} redirects",
                request=response.request,
            )

        return response, redirects_followed, str(hostname_url)
