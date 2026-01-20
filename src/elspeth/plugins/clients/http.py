# src/elspeth/plugins/clients/http.py
"""Audited HTTP client with automatic call recording."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.clients.base import AuditedClientBase

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class AuditedHTTPClient(AuditedClientBase):
    """HTTP client that automatically records all calls to audit trail.

    Wraps httpx to ensure every HTTP call is recorded to the Landscape
    audit trail. Supports:
    - Automatic request/response recording
    - Auth header filtering (not recorded for security)
    - Latency measurement
    - Error recording

    Example:
        client = AuditedHTTPClient(
            recorder=recorder,
            state_id=state_id,
            base_url="https://api.example.com",
            headers={"Authorization": "Bearer ..."},
        )

        response = client.post("/v1/process", json={"data": "value"})
        print(response.json())
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        state_id: str,
        *,
        timeout: float = 30.0,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize audited HTTP client.

        Args:
            recorder: LandscapeRecorder for audit trail storage
            state_id: Node state ID to associate calls with
            timeout: Request timeout in seconds (default: 30.0)
            base_url: Optional base URL to prepend to all requests
            headers: Default headers for all requests
        """
        super().__init__(recorder, state_id)
        self._timeout = timeout
        self._base_url = base_url
        self._default_headers = headers or {}

    # Headers that may contain secrets - excluded from audit trail
    _SENSITIVE_REQUEST_HEADERS = frozenset(
        {"authorization", "x-api-key", "api-key", "x-auth-token", "proxy-authorization"}
    )
    _SENSITIVE_RESPONSE_HEADERS = frozenset(
        {"set-cookie", "www-authenticate", "proxy-authenticate", "x-auth-token"}
    )

    def _filter_request_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Filter out sensitive request headers from audit recording.

        Auth headers are not recorded to avoid storing secrets.

        Args:
            headers: Full headers dict

        Returns:
            Headers dict with sensitive headers removed
        """
        return {
            k: v
            for k, v in headers.items()
            if k.lower() not in self._SENSITIVE_REQUEST_HEADERS
            and "auth" not in k.lower()
            and "key" not in k.lower()
            and "secret" not in k.lower()
            and "token" not in k.lower()
        }

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
        call_index = self._next_call_index()

        full_url = f"{self._base_url}{url}" if self._base_url else url
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
            # Try to decode as JSON for structured storage, fall back to text
            response_body: Any = None
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    response_body = response.json()
                except Exception:
                    # If JSON decode fails, store as text
                    response_body = response.text
            else:
                # For non-JSON, store text (truncated for very large responses)
                response_body = (
                    response.text[:100_000]
                    if len(response.text) > 100_000
                    else response.text
                )

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
            raise
