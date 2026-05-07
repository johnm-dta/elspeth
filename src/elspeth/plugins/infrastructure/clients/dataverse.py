"""Dataverse OData v4 REST API client.

Pure protocol client for Microsoft Dataverse. Handles authentication,
pagination (OData nextLink and FetchXML paging cookie), rate limiting,
error classification, and SSRF validation. Does NOT record audit calls —
the source/sink plugin is responsible for calling ctx.record_call() after
each HTTP call.
"""

from __future__ import annotations

import fnmatch
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import httpx
import structlog
from pydantic import BaseModel, Field, model_validator

from elspeth.contracts.freeze import freeze_fields
from elspeth.core.security.web import NetworkError, SSRFBlockedError, SSRFSafeRequest, validate_url_for_ssrf
from elspeth.plugins.infrastructure.clients.fingerprinting import (
    filter_response_headers,
    fingerprint_headers,
)
from elspeth.plugins.infrastructure.clients.json_utils import parse_json_strict

if TYPE_CHECKING:
    from typing import Self

    from azure.identity import ClientSecretCredential, DefaultAzureCredential, ManagedIdentityCredential

    from elspeth.core.rate_limit.limiter import RateLimiter
    from elspeth.core.rate_limit.registry import NoOpLimiter

logger = structlog.get_logger(__name__)


# Microsoft-documented Dataverse environment domain patterns.
# Used for SSRF pre-filtering before IP-pinning validation.
DATAVERSE_DOMAIN_ALLOWLIST = (
    "*.crm.dynamics.com",  # Commercial cloud (default)
    "*.crm2.dynamics.com",  # Regional commercial instances
    "*.crm3.dynamics.com",
    "*.crm4.dynamics.com",
    "*.crm5.dynamics.com",
    "*.crm6.dynamics.com",
    "*.crm7.dynamics.com",
    "*.crm8.dynamics.com",
    "*.crm9.dynamics.com",  # US Government (GCC)
    "*.crm10.dynamics.com",
    "*.crm11.dynamics.com",
    "*.crm.microsoftdynamics.us",  # US Government (GCC High)
    "*.crm.appsplatform.us",  # US Government (DoD)
    "*.crm.microsoftdynamics.de",  # Germany (legacy)
    "*.crm.dynamics.cn",  # China (21Vianet)
)

# Regex for validating additional domain patterns.
# Must target a legitimate Microsoft sovereign cloud TLD with valid hostname labels.
_ADDITIONAL_DOMAIN_PATTERN = re.compile(
    r"^(\*\.)?([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+"
    r"(dynamics\.com|dynamics\.cn|microsoftdynamics\.(us|de)|appsplatform\.us)$"
)

# Maximum consecutive empty pages before pagination terminates.
_MAX_CONSECUTIVE_EMPTY_PAGES = 3


class DataverseClientError(Exception):
    """Exception for Dataverse protocol-level errors.

    Raised by DataverseClient for HTTP errors, JSON parse failures, and
    SSRF validation rejections. The source/sink plugin catches these to
    record audit entries via ctx.record_call() before re-raising for
    engine retry/quarantine handling.

    error_category is a typed classification set at the raise site,
    enabling structured audit recording without fragile string-matching.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        latency_ms: float | None = None,
        error_category: str = "protocol_error",
        request_url: str | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.latency_ms = latency_ms
        self.error_category = error_category
        self.request_url = request_url
        self.request_headers = request_headers


@dataclass(frozen=True, slots=True)
class DataversePageResponse:
    """Validated page response from Dataverse.

    Constructed by DataverseClient after Tier 3 validation (JSON parse,
    NaN/Infinity rejection, structure validation). The source/sink plugin
    consumes this as Tier 2 data.
    """

    status_code: int
    rows: list[dict[str, Any]]
    latency_ms: float
    headers: dict[str, str]
    request_headers: dict[str, str]  # Headers actually sent in the request (already fingerprinted)
    request_url: str  # URL that was actually fetched (for accurate audit trail)
    next_link: str | None  # @odata.nextLink URL, if present (structured queries)
    paging_cookie: str | None  # FetchXML paging cookie, if present
    more_records: bool | None  # True/False if Dataverse stated it; None if field absent

    def __post_init__(self) -> None:
        if self.next_link is not None and self.paging_cookie is not None:
            raise ValueError(
                "next_link and paging_cookie are mutually exclusive — OData queries use next_link, FetchXML queries use paging_cookie"
            )
        freeze_fields(self, "rows", "headers", "request_headers")


class DataverseAuthConfig(BaseModel):
    """Authentication configuration for Dataverse connections."""

    model_config = {"extra": "forbid", "frozen": True}

    method: Literal["service_principal", "managed_identity"]

    # Service principal fields (required when method=service_principal)
    # repr=False on client_secret prevents credential exposure in stack traces
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_auth_fields(self) -> Self:
        if self.method == "service_principal":
            missing = []
            if not self.tenant_id or not self.tenant_id.strip():
                missing.append("tenant_id")
            if not self.client_id or not self.client_id.strip():
                missing.append("client_id")
            if not self.client_secret or not self.client_secret.strip():
                missing.append("client_secret")
            if missing:
                raise ValueError(f"service_principal auth requires: {', '.join(missing)}")
        return self

    def create_credential(self) -> ClientSecretCredential | ManagedIdentityCredential:
        """Construct an azure-identity credential from this config.

        Returns the appropriate credential type based on the configured
        auth method. Fields are guaranteed non-None by model_validator
        for service_principal mode.

        Returns:
            ClientSecretCredential or ManagedIdentityCredential

        Raises:
            ImportError: If azure-identity is not installed.
        """
        from azure.identity import ClientSecretCredential, ManagedIdentityCredential

        if self.method == "service_principal":
            if self.tenant_id is None or self.client_id is None or self.client_secret is None:
                missing = [f for f in ("tenant_id", "client_id", "client_secret") if getattr(self, f) is None]
                raise RuntimeError(
                    f"service_principal auth fields are None at credential creation time: {missing}. "
                    f"model_validator should have rejected this at construction — this is a bug."
                )
            return ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
        return ManagedIdentityCredential()


def _validate_domain_allowlist(hostname: str, additional_domains: tuple[str, ...] = ()) -> bool:
    """Validate hostname against Dataverse domain allowlist.

    Uses fnmatch on the hostname component only (not full URL).
    fnmatch applies implicit anchoring — pattern must match the entire
    hostname string, preventing suffix injection attacks.

    Args:
        hostname: Hostname to validate
        additional_domains: Extra domain patterns (append-only)

    Returns:
        True if hostname matches any allowed pattern
    """
    all_patterns = DATAVERSE_DOMAIN_ALLOWLIST + additional_domains
    return any(fnmatch.fnmatch(hostname, pattern) for pattern in all_patterns)


def validate_additional_domain(pattern: str) -> None:
    """Validate an additional domain pattern for safety.

    Rejects patterns that don't target a legitimate Microsoft sovereign
    cloud TLD.

    Args:
        pattern: Domain pattern to validate

    Raises:
        ValueError: If pattern doesn't match the safety guard regex
    """
    if not _ADDITIONAL_DOMAIN_PATTERN.match(pattern):
        raise ValueError(
            f"Additional domain pattern {pattern!r} rejected: must target a Microsoft "
            f"sovereign cloud TLD (dynamics.com, dynamics.cn, microsoftdynamics.us, "
            f"microsoftdynamics.de, appsplatform.us) with valid hostname labels."
        )


class DataverseClient:
    """OData v4 REST API client for Microsoft Dataverse.

    Handles protocol-level concerns:
    - Authentication via azure-identity (service principal or managed identity)
    - Pagination (OData nextLink and FetchXML paging cookie)
    - Rate limiting via shared RateLimitRegistry
    - Error classification (retryable vs non-retryable)
    - SSRF validation on all URLs (domain allowlist + IP-pinning)
    - Strict JSON parsing (NaN/Infinity rejection at Tier 3 boundary)

    Does NOT record audit calls — returns response data to the caller.
    The source/sink plugin calls ctx.record_call() after each HTTP call.
    """

    def __init__(
        self,
        *,
        environment_url: str,
        credential: ClientSecretCredential | DefaultAzureCredential | ManagedIdentityCredential,
        api_version: str = "v9.2",
        limiter: RateLimiter | NoOpLimiter | None = None,
        retry_after_cap: float = 60.0,
        additional_domains: tuple[str, ...] = (),
    ) -> None:
        self._environment_url = environment_url.rstrip("/")
        self._credential = credential
        self._api_version = api_version
        self._limiter = limiter
        self._retry_after_cap = retry_after_cap
        self._additional_domains = additional_domains

        # Validate environment_url against domain allowlist
        parsed = urllib.parse.urlparse(self._environment_url)
        hostname = parsed.hostname
        if hostname is None:
            raise DataverseClientError(
                f"Cannot extract hostname from environment_url: {environment_url!r}",
                retryable=False,
            )
        if not _validate_domain_allowlist(hostname, self._additional_domains):
            raise DataverseClientError(
                f"environment_url hostname {hostname!r} does not match any Dataverse "
                f"domain pattern. Allowed patterns: {DATAVERSE_DOMAIN_ALLOWLIST}",
                retryable=False,
            )

        # Token scope for Dataverse
        self._token_scope = f"{self._environment_url}/.default"

        # httpx.Client with connection pooling. follow_redirects=False per spec:
        # redirects bypass the two-layer SSRF validation.
        self._client = httpx.Client(
            timeout=30.0,
            follow_redirects=False,
        )

        # Instance-level flag: credential reconstruction is allowed at most
        # once per client lifetime. A second 401 after reconstruction means the
        # credentials themselves are invalid, not just an expired token.
        # Intentionally NOT reset per-pagination — prevents infinite auth loops
        # across long multi-page fetches.
        self._auth_retried = False

    def _acquire_rate_limit(self) -> None:
        """Acquire rate limit before each request."""
        if self._limiter is not None:
            self._limiter.acquire()

    def get_auth_headers(self) -> dict[str, str]:
        """Get authorization headers with current token.

        Public API: callers (source/sink plugins) need this for audit
        fingerprinting via fingerprint_headers(). azure-identity handles
        caching and transparent refresh internally.
        """
        token = self._credential.get_token(self._token_scope)
        return {
            "Authorization": f"Bearer {token.token}",
            "Accept": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
        }

    def _validate_url_ssrf(self, url: str) -> SSRFSafeRequest:
        """Two-layer SSRF validation: domain allowlist + IP-pinning.

        Returns an SSRFSafeRequest with the resolved IP pinned. Callers
        MUST use safe_request.connection_url for the actual HTTP call
        and pass safe_request.host_header as the Host header. This
        prevents DNS rebinding between validation and connection.

        Args:
            url: URL to validate

        Returns:
            SSRFSafeRequest with pinned IP for direct connection

        Raises:
            DataverseClientError: If URL fails either validation layer
        """
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname

        if hostname is None:
            raise DataverseClientError(
                f"Cannot extract hostname from URL: {url!r}",
                retryable=False,
            )

        # Layer 1: Domain allowlist pre-filter
        if not _validate_domain_allowlist(hostname, self._additional_domains):
            raise DataverseClientError(
                f"URL hostname {hostname!r} rejected by domain allowlist. Possible SSRF attempt via @odata.nextLink redirection.",
                retryable=False,
                error_category="ssrf_rejected",
            )

        # Layer 2: IP-pinning validation (prevents DNS rebinding)
        try:
            return validate_url_for_ssrf(url)
        except (SSRFBlockedError, NetworkError) as exc:
            raise DataverseClientError(
                f"URL {url!r} failed IP-pinning SSRF validation: {exc}",
                retryable=False,
                error_category="ssrf_rejected",
            ) from exc

    def _classify_error(
        self,
        status_code: int,
        headers: dict[str, str],
        latency_ms: float,
        *,
        request_url: str | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> DataverseClientError:
        """Classify HTTP error by status code.

        Returns a DataverseClientError with retryable flag set appropriately.
        """
        # 429: Rate limited — check Retry-After
        if status_code == 429:
            retry_after_raw = headers.get("retry-after", "")
            try:
                retry_after = float(retry_after_raw)
            except (ValueError, TypeError):
                retry_after = 1.0

            # Clamp for classification: floor=1, ceiling=retry_after_cap
            effective = max(1.0, min(retry_after, self._retry_after_cap))

            # If server requests longer than cap, classify as non-retryable
            if retry_after > self._retry_after_cap:
                return DataverseClientError(
                    f"Rate limited with Retry-After {retry_after}s exceeding cap "
                    f"of {self._retry_after_cap}s — classifying as non-retryable "
                    f"(server signaling extended unavailability)",
                    retryable=False,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    request_url=request_url,
                    request_headers=request_headers,
                )
            return DataverseClientError(
                f"Rate limited (429) with Retry-After {effective}s",
                retryable=True,
                status_code=status_code,
                latency_ms=latency_ms,
                request_url=request_url,
                request_headers=request_headers,
            )

        # 401: Auth failure — retryable once via credential reconstruction
        if status_code == 401:
            return DataverseClientError(
                "Authentication failed (401) — token may be expired",
                retryable=not self._auth_retried,
                status_code=status_code,
                latency_ms=latency_ms,
                error_category="auth_failure",
                request_url=request_url,
                request_headers=request_headers,
            )

        # Non-retryable client errors
        if status_code in (400, 403, 404, 409, 412):
            labels = {
                400: "Bad request",
                403: "Forbidden",
                404: "Not found",
                409: "Conflict (duplicate)",
                412: "Precondition failed (optimistic concurrency)",
            }
            return DataverseClientError(
                f"{labels[status_code]} ({status_code})",
                retryable=False,
                status_code=status_code,
                latency_ms=latency_ms,
                request_url=request_url,
                request_headers=request_headers,
            )

        # 5xx: Retryable server errors
        if 500 <= status_code < 600:
            return DataverseClientError(
                f"Server error ({status_code})",
                retryable=True,
                status_code=status_code,
                latency_ms=latency_ms,
                request_url=request_url,
                request_headers=request_headers,
            )

        # Any other error status
        return DataverseClientError(
            f"Unexpected HTTP status {status_code}",
            retryable=False,
            status_code=status_code,
            latency_ms=latency_ms,
            request_url=request_url,
            request_headers=request_headers,
        )

    def _execute_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        ssrf_safe: SSRFSafeRequest | None = None,
    ) -> DataversePageResponse:
        """Execute an HTTP request against Dataverse.

        Handles auth header injection, rate limiting, response parsing,
        and error classification. Returns a validated DataversePageResponse.

        When ssrf_safe is provided, the request connects to the pre-validated
        IP address (preventing DNS rebinding) while preserving the original
        hostname via the Host header for virtual hosting and TLS SNI.

        Args:
            method: HTTP method (GET or PATCH)
            url: Full URL to request (used for audit recording)
            json_body: JSON body for PATCH requests
            extra_headers: Additional headers to merge
            ssrf_safe: Pre-validated SSRFSafeRequest with pinned IP.
                When provided, the actual HTTP call uses connection_url
                (IP-based) instead of url (hostname-based).

        Returns:
            DataversePageResponse with validated response data

        Raises:
            DataverseClientError: For protocol-level errors
        """
        self._acquire_rate_limit()

        auth_headers = self.get_auth_headers()
        headers = {**auth_headers, **(extra_headers or {})}
        # Fingerprint auth headers early so every error path can carry
        # request metadata for accurate audit trail recording.
        fingerprinted = fingerprint_headers(auth_headers)

        # IP-pinning SSRF validation on every outbound request.
        # When ssrf_safe is pre-validated (e.g., from paginate_odata nextLink
        # handling), reuse it. Otherwise, validate now — even for config-derived
        # URLs, because DNS rebinding can occur between __init__ domain check
        # and the actual TCP connect.
        if ssrf_safe is None:
            ssrf_safe = self._validate_url_ssrf(url)

        # Connect to the pinned IP and set the Host header for virtual hosting.
        # The sni_hostname extension tells httpx/httpcore to use the original
        # domain for TLS SNI and certificate verification instead of the IP literal.
        extensions: dict[str, bytes] | None = None
        connection_url = ssrf_safe.connection_url
        headers["Host"] = ssrf_safe.host_header
        if ssrf_safe.scheme == "https":
            extensions = {"sni_hostname": ssrf_safe.sni_hostname.encode("ascii")}

        start = time.perf_counter()
        try:
            if method == "PATCH":
                response = self._client.patch(
                    connection_url,
                    json=json_body,
                    headers=headers,
                    extensions=extensions,
                )
            else:
                response = self._client.get(connection_url, headers=headers, extensions=extensions)
        except httpx.TimeoutException as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            raise DataverseClientError(
                f"Request timed out: {exc}",
                retryable=True,
                latency_ms=latency_ms,
                request_url=url,
                request_headers=fingerprinted,
            ) from exc
        except httpx.ConnectError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            raise DataverseClientError(
                f"Connection failed: {exc}",
                retryable=True,
                latency_ms=latency_ms,
                request_url=url,
                request_headers=fingerprinted,
            ) from exc
        except httpx.HTTPError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            raise DataverseClientError(
                f"HTTP error: {exc}",
                retryable=True,
                latency_ms=latency_ms,
                request_url=url,
                request_headers=fingerprinted,
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        # Filter sensitive headers (Set-Cookie, WWW-Authenticate, etc.) at
        # the DTO boundary — same pattern as AuditedHTTPClient.
        resp_headers = filter_response_headers(dict(response.headers))

        # Handle redirects — reject, don't follow
        if 300 <= response.status_code < 400:
            raise DataverseClientError(
                f"Redirect response ({response.status_code}) from Dataverse — "
                f"redirects are not followed (SSRF protection). "
                f"Location: {resp_headers.get('location', 'absent')}",
                retryable=False,
                status_code=response.status_code,
                latency_ms=latency_ms,
                request_url=url,
                request_headers=fingerprinted,
            )

        # Non-success status → classify error
        if response.status_code < 200 or response.status_code >= 300:
            raise self._classify_error(
                response.status_code,
                resp_headers,
                latency_ms,
                request_url=url,
                request_headers=fingerprinted,
            )

        # For PATCH responses that return 204 No Content
        if response.status_code == 204 or not response.text:
            return DataversePageResponse(
                status_code=response.status_code,
                rows=[],
                latency_ms=latency_ms,
                headers=resp_headers,
                request_headers=fingerprinted,
                request_url=url,
                next_link=None,
                paging_cookie=None,
                more_records=None,  # No body → no morerecords field to record
            )

        # Tier 3 boundary: parse JSON strictly
        parsed, error = parse_json_strict(response.text)
        if error is not None:
            raise DataverseClientError(
                f"Invalid JSON from Dataverse: {error}",
                retryable=False,
                status_code=response.status_code,
                latency_ms=latency_ms,
                request_url=url,
                request_headers=fingerprinted,
            )

        # Validate response structure
        if not isinstance(parsed, dict):
            raise DataverseClientError(
                f"Expected JSON object from Dataverse, got {type(parsed).__name__}",
                retryable=False,
                status_code=response.status_code,
                latency_ms=latency_ms,
                request_url=url,
                request_headers=fingerprinted,
            )

        # Extract rows from "value" array (queries) or treat as single-item
        rows: list[dict[str, Any]]
        if "value" in parsed:
            value = parsed["value"]
            if not isinstance(value, list):
                raise DataverseClientError(
                    f"Expected 'value' to be an array, got {type(value).__name__}",
                    retryable=False,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    request_url=url,
                    request_headers=fingerprinted,
                )
            # Tier 3 boundary: validate each item is a dict. Non-dict elements
            # (strings, ints, nulls) in the value array are malformed responses
            # from Dataverse that would bypass downstream row processing.
            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    raise DataverseClientError(
                        f"Expected 'value[{i}]' to be a JSON object, got {type(item).__name__}",
                        retryable=False,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                        request_url=url,
                        request_headers=fingerprinted,
                    )
            rows = value
        else:
            # Single record response (e.g., PATCH upsert returns the record)
            rows = [parsed]

        # Extract pagination metadata
        next_link = parsed.get("@odata.nextLink")
        if next_link is not None and not isinstance(next_link, str):
            raise DataverseClientError(
                f"Expected '@odata.nextLink' to be a string, got {type(next_link).__name__}",
                retryable=False,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )

        # FetchXML paging cookie
        paging_cookie = parsed.get("@Microsoft.Dynamics.CRM.fetchxmlpagingcookie")
        if paging_cookie is not None and not isinstance(paging_cookie, str):
            raise DataverseClientError(
                f"Expected '@Microsoft.Dynamics.CRM.fetchxmlpagingcookie' to be a string, got {type(paging_cookie).__name__}",
                retryable=False,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )

        more_records_raw = parsed.get("@Microsoft.Dynamics.CRM.morerecords")
        if more_records_raw is not None:
            # Tier 3 boundary: Dataverse returns morerecords as string "true"/"false"
            # or bool. bool("false") == True in Python — must parse explicitly.
            if isinstance(more_records_raw, bool):
                more_records = more_records_raw
            elif isinstance(more_records_raw, str):
                lower = more_records_raw.lower()
                if lower == "true":
                    more_records = True
                elif lower == "false":
                    more_records = False
                else:
                    raise DataverseClientError(
                        f"'@Microsoft.Dynamics.CRM.morerecords' has unexpected string value "
                        f"{more_records_raw!r} — expected 'true' or 'false'",
                        retryable=False,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                    )
            else:
                raise DataverseClientError(
                    f"'@Microsoft.Dynamics.CRM.morerecords' has unexpected type "
                    f"{type(more_records_raw).__name__} — expected bool or string",
                    retryable=False,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                )
        else:
            # Field absent — record the absence, don't infer.
            # OData queries don't include morerecords (uses nextLink instead).
            # FetchXML queries should always include it — absence there is
            # an anomaly the consumer should detect, not one we paper over.
            more_records = None

        return DataversePageResponse(
            status_code=response.status_code,
            rows=rows,
            latency_ms=latency_ms,
            headers=resp_headers,
            request_headers=fingerprinted,
            request_url=url,
            next_link=next_link,
            paging_cookie=paging_cookie if next_link is None else None,
            more_records=more_records,
        )

    def get_page(self, url: str) -> DataversePageResponse:
        """Fetch a single page of data from Dataverse.

        Args:
            url: Full URL to fetch

        Returns:
            DataversePageResponse with validated page data

        Raises:
            DataverseClientError: For protocol-level errors
        """
        return self._execute_request("GET", url)

    def upsert(self, url: str, body: dict[str, Any]) -> DataversePageResponse:
        """PATCH upsert a record to Dataverse.

        Args:
            url: Full URL including entity and key (e.g., /api/data/v9.2/contacts(key=val))
            body: JSON body to send

        Returns:
            DataversePageResponse (typically 204 with empty rows)

        Raises:
            DataverseClientError: For protocol-level errors
        """
        return self._execute_request("PATCH", url, json_body=body)

    def paginate_odata(self, initial_url: str) -> Iterator[DataversePageResponse]:
        """Paginate through structured OData query results.

        Follows @odata.nextLink URLs with SSRF validation and IP-pinning
        at each hop. Terminates when no nextLink is present or after
        consecutive empty pages (empty-page guard).

        Args:
            initial_url: First page URL

        Yields:
            DataversePageResponse per page

        Raises:
            DataverseClientError: On SSRF validation failure, empty-page guard,
                or protocol errors
        """
        url = initial_url
        ssrf_safe: SSRFSafeRequest | None = None
        consecutive_empty = 0

        while True:
            page = self._execute_request("GET", url, ssrf_safe=ssrf_safe)
            yield page

            # Empty-page guard
            if not page.rows:
                consecutive_empty += 1
                if consecutive_empty >= _MAX_CONSECUTIVE_EMPTY_PAGES:
                    raise DataverseClientError(
                        f"Pagination terminated: {_MAX_CONSECUTIVE_EMPTY_PAGES} consecutive "
                        f"empty pages received from Dataverse. This may indicate a server-side "
                        f"edge condition producing nextLink URLs with no data.",
                        retryable=False,
                        error_category="empty_page_guard",
                    )
            else:
                consecutive_empty = 0

            # No more pages
            if page.next_link is None:
                break

            # SSRF validate the nextLink and pin the resolved IP.
            # The returned SSRFSafeRequest is passed to _execute_request
            # so the HTTP call connects to the validated IP, preventing
            # DNS rebinding between validation and connection.
            ssrf_safe = self._validate_url_ssrf(page.next_link)
            url = page.next_link

    def paginate_fetchxml(self, entity: str, fetch_xml: str) -> Iterator[DataversePageResponse]:
        """Paginate through FetchXML query results.

        Uses paging cookie mechanism: injects cookie and page number into
        the <fetch> element via xml.etree.ElementTree (XML-safe injection).
        Parses the XML once upfront and mutates the ET root between pages to
        avoid re-parsing on every iteration.

        Args:
            entity: Entity logical name for URL construction
            fetch_xml: FetchXML query string

        Yields:
            DataversePageResponse per page

        Raises:
            DataverseClientError: On XML parse error, root element validation
                failure, or protocol errors
        """
        # Parse once — validated at config time, but guard here for safety
        try:
            root = ET.fromstring(fetch_xml)
        except ET.ParseError as exc:
            raise DataverseClientError(
                f"Failed to parse FetchXML for pagination: {exc}",
                retryable=False,
            ) from exc

        if root.tag != "fetch":
            raise DataverseClientError(
                f"FetchXML root element must be <fetch>, got <{root.tag}>. Paging cookie injection requires a valid FetchXML structure.",
                retryable=False,
            )

        current_page = 1

        while True:
            xml_str = ET.tostring(root, encoding="unicode")
            encoded_xml = urllib.parse.quote(xml_str)
            url = f"{self._environment_url}/api/data/{self._api_version}/{entity}?fetchXml={encoded_xml}"

            page = self.get_page(url)
            yield page

            # Check if more records exist.
            # FetchXML responses must include morerecords — its absence is a
            # protocol anomaly we must not silently infer around.
            if page.more_records is None:
                raise DataverseClientError(
                    "FetchXML response missing '@Microsoft.Dynamics.CRM.morerecords' field "
                    "— cannot determine pagination state. OData uses nextLink instead, "
                    "but FetchXML pagination requires this field.",
                    retryable=False,
                    error_category="protocol_violation",
                )
            if not page.more_records:
                break

            # Tier 3 boundary: morerecords=True with no paging cookie is a
            # contradictory signal — Dataverse claims more data exists but
            # provides no mechanism to retrieve it. Surface as protocol error
            # rather than silently truncating the result set.
            if page.paging_cookie is None:
                raise DataverseClientError(
                    "FetchXML pagination: morerecords=True but no paging cookie provided. "
                    "Cannot retrieve remaining pages — refusing to silently truncate results.",
                    retryable=False,
                    error_category="protocol_violation",
                )

            # Inject paging cookie into the existing ET root.
            # The paging cookie is Tier 3 data — ET.set() handles
            # attribute escaping automatically (no string formatting).
            decoded_cookie = urllib.parse.unquote(page.paging_cookie)

            current_page += 1
            root.set("paging-cookie", decoded_cookie)
            root.set("page", str(current_page))

    def reconstruct_credential(
        self,
        auth_config: DataverseAuthConfig,
    ) -> None:
        """Reconstruct the credential for 401 retry.

        azure-identity's get_token() does not expose a force_refresh parameter.
        The only reliable way to force re-authentication is credential reconstruction.

        Args:
            auth_config: Auth configuration to rebuild from
        """
        self._credential = auth_config.create_credential()
        self._auth_retried = True

    def close(self) -> None:
        """Release HTTP client resources."""
        self._client.close()
