# Dataverse Source and Sink Plugins — Design Spec

**Date:** 2026-03-18
**Status:** Reviewed (R7 — R6 issues resolved; R7 addresses panel review findings: HTTPS enforcement, parse_json_strict error handling, FetchXML root validation, extraction branch coverage, rate_limit_registry null guard, schema lock scoping, SSRF audit recording, auth config rationale, ArtifactDescriptor factory)
**Scope:** New `dataverse` source plugin, new `dataverse` sink plugin (upsert-only day-one), shared `DataverseClient` infrastructure

## Overview

Add Microsoft Dataverse integration to ELSPETH via a source plugin (read entities) and sink plugin (write/upsert entities). Dataverse exposes data through an OData v4 REST API secured by Azure Entra ID. Both plugins share a `DataverseClient` that handles protocol-level concerns (auth, pagination, rate limiting, error classification). Audit recording uses the standard `ctx.record_call()` pattern — the source/sink plugin calls `SourceContext.record_call()` / `SinkContext.record_call()` per the `AzureBlobSource` pattern, not `LandscapeRecorder.record_call()` directly.

## Motivation

Dataverse is the data platform underlying Microsoft Dynamics 365, Power Platform, and many enterprise systems. Pipelines that sense from or act upon Dataverse data need first-class support — not ad-hoc HTTP transforms that would bypass audit recording, schema contracts, and field normalization.

## Architecture

### File Layout

```
src/elspeth/plugins/
├── infrastructure/
│   └── clients/
│       ├── dataverse.py          # DataverseClient
│       ├── fingerprinting.py     # Shared HMAC header fingerprinting (extracted from http.py)
│       └── json_utils.py         # Shared strict JSON parsing (extracted from http.py)
├── sources/
│   └── dataverse.py              # DataverseSource + DataverseSourceConfig
└── sinks/
    └── dataverse.py              # DataverseSink + DataverseSinkConfig
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `DataverseClient` | OData protocol, auth, pagination, rate limiting, error classification. Returns response data to caller; does NOT record audit calls directly. |
| `DataverseSource` | Entity query config, row streaming, field normalization, schema contracts |
| `DataverseSink` | Write mode dispatch, field mapping, lookup binding, batch accumulation |

### Layer Placement

All three components live at L3 (plugins). `DataverseClient` sits in `plugins/infrastructure/clients/` alongside the existing `AuditedHTTPClient` and `AuditedLLMClient`. No new cross-layer dependencies are introduced. `DataverseClient` does NOT wrap `AuditedHTTPClient` — it uses `httpx.Client` directly for HTTP calls and returns response data to the caller. The source/sink plugin then calls `ctx.record_call()` to record each call in the audit trail (see "HTTP and Audit Recording Pattern" section).

## DataverseClient

### HTTP and Audit Recording Pattern

`DataverseClient` uses `httpx.Client` directly for HTTP calls and **returns response data to the caller**. It does NOT record audit calls itself. The source/sink plugin is responsible for calling `ctx.record_call()` (`SourceContext.record_call()` / `SinkContext.record_call()`) after each HTTP call, following the `AzureBlobSource` pattern at `azure_blob_source.py:434`.

**Why this pattern:** Sources and sinks do not have access to `state_id` (a per-row concept on `TransformContext`), so they cannot construct an `AuditedHTTPClient`. The context-level `record_call()` method abstracts away the `state_id`/`call_index` allocation — the plugin supplies `(call_type, status, request_data, response_data, latency_ms, *, provider)` and the context handles the rest.

**Call flow (success path):**
```python
# In DataverseSource.load() or DataverseSink.write():
response = self._client.get_page(url)  # DataverseClient returns response data
ctx.record_call(
    call_type=CallType.HTTP,
    status=CallStatus.SUCCESS,
    request_data={"method": "GET", "url": url, "headers": fingerprinted_headers},
    response_data={"status_code": response.status_code, "row_count": len(response.rows)},
    latency_ms=response.latency_ms,
    provider="dataverse",
)
```

**Call flow (error path — MANDATORY):**
```python
# Every failed HTTP call MUST also be recorded — per CLAUDE.md: "if it's not recorded, it didn't happen"
# Follows AzureBlobSource error recording pattern at azure_blob_source.py:455
try:
    response = self._client.get_page(url)
except DataverseClientError as exc:
    ctx.record_call(
        call_type=CallType.HTTP,
        status=CallStatus.ERROR,
        request_data={"method": "GET", "url": url, "headers": fingerprinted_headers},
        response_data=None,
        error={"error_type": type(exc).__name__, "message": str(exc), "status_code": exc.status_code},
        latency_ms=exc.latency_ms,
        provider="dataverse",
    )
    raise  # Re-raise for engine retry/quarantine handling
```

Each recorded call captures:
- Request method, URL, headers (with secret fingerprinting via shared `fingerprint_headers()` utility — see below)
- Response status, latency, row count per page (success) or error details (failure)
- Telemetry emission (`ExternalCallCompleted` events) via the `telemetry_emit` callback stored in `on_start()` from `LifecycleContext`. **Note:** `SinkContext` does not expose `telemetry_emit` directly. The sink stores `self._telemetry_emit = ctx.telemetry_emit` during `on_start(ctx: LifecycleContext)` and uses this stored callback during `write()` to emit per-row telemetry events. This is the same lifecycle-context-capture pattern used for `run_id` and `rate_limit_registry`.

`DataverseClient` adds the OData-specific protocol layer on top (auth, pagination, error classification) but is audit-unaware — it is a pure protocol client.

**Redirect policy:** `DataverseClient` MUST construct its `httpx.Client` with `follow_redirects=False`. HTTP redirects are not followed automatically — any redirect from a Dataverse endpoint would bypass the two-layer SSRF validation (domain allowlist + IP-pinning). If Dataverse returns a redirect response (3xx), the client treats it as a non-retryable error.

**Header fingerprinting (MANDATORY):** Because `DataverseClient` bypasses `AuditedHTTPClient`, it does not get automatic HMAC fingerprinting of sensitive headers (`Authorization: Bearer <token>`, `ocp-apim-subscription-key`, etc.). The source/sink plugin MUST fingerprint headers before passing them to `ctx.record_call()` — otherwise, bearer tokens are written to the Landscape `calls` table in plaintext.

The fingerprinting logic (`_is_sensitive_header()`, `_filter_request_headers()`, `SENSITIVE_HEADERS_EXACT`, `SENSITIVE_HEADER_WORDS`) is extracted from `AuditedHTTPClient` into a shared utility at `plugins/infrastructure/clients/fingerprinting.py`. Both `AuditedHTTPClient` and `DataverseClient` callers import from this shared module. The shared utility also enforces the `ELSPETH_FINGERPRINT_KEY` presence check (raising `FrameworkBugError` when absent in non-dev mode), preserving the same backstop that `AuditedHTTPClient` provides.

```python
from elspeth.plugins.infrastructure.clients.fingerprinting import fingerprint_headers

fingerprinted = fingerprint_headers(raw_headers)  # HMAC-fingerprints sensitive values
ctx.record_call(
    ...,
    request_data={"method": "GET", "url": url, "headers": fingerprinted},
    ...
)
```

### Fingerprinting and JSON Utility Extraction (Prerequisite)

The shared utilities `fingerprinting.py` and `json_utils.py` are extracted from `plugins/infrastructure/clients/http.py` (`AuditedHTTPClient`) and MUST be delivered as a **separate commit before the Dataverse plugins**, with its own CI pass. This extraction is a prerequisite, not part of the Dataverse plugin work.

**Extraction mandate:**

1. **Create `plugins/infrastructure/clients/fingerprinting.py`** containing `fingerprint_headers()`, `_is_sensitive_header()`, `_filter_request_headers()`, `SENSITIVE_HEADERS_EXACT`, `SENSITIVE_HEADER_WORDS`, and the `ELSPETH_FINGERPRINT_KEY` presence check.
2. **Create `plugins/infrastructure/clients/json_utils.py`** containing `parse_json_strict()`, `_contains_non_finite()`, and related constants.
3. **Delete the private copies** (`_contains_non_finite`, `_parse_json_strict`, `_is_sensitive_header`, `_filter_request_headers`, and their associated constants) from `http.py`.
4. **Update `AuditedHTTPClient`** in `http.py` to import from the new shared modules instead of using its own private copies.
5. **Add a regression test** that asserts byte-identical output between the extracted utilities and the original methods. The test MUST cover **all branches exhaustively** (this is security-critical code, not a candidate for representative sampling):
   - **`fingerprint_headers()`**: (a) no fingerprint key + non-dev mode → `FrameworkBugError`, (b) no fingerprint key + dev mode → header removed, (c) fingerprint key present → HMAC fingerprint applied
   - **`_is_sensitive_header()`**: (a) exact match (`authorization` → sensitive), (b) word match (`ocp-apim-subscription-key` → sensitive), (c) `x-`prefix match (`x-secret` → sensitive, `x-author` → not sensitive), (d) non-sensitive header passthrough
   - **`parse_json_strict()`**: (a) valid JSON → parsed, (b) malformed JSON → error tuple, (c) NaN in nested object → rejected, (d) Infinity in array → rejected

   **Note on call convention change:** `_is_sensitive_header()` and `_filter_request_headers()` are currently **instance methods** on `AuditedHTTPClient` (using `self._SENSITIVE_HEADER_WORDS`, etc.). The extraction promotes them to module-level functions with module-level constants. The `self` parameter is removed and `self._SENSITIVE_HEADER_WORDS` becomes `SENSITIVE_HEADER_WORDS` at module scope. The regression test guards against subtle behavioral drift during this promotion.

This extraction-then-delete pattern ensures no duplication — the shared utilities are the single source of truth for both `AuditedHTTPClient` and `DataverseClient` callers.

### Authentication

Two methods, both producing OAuth2 bearer tokens against the Dataverse environment URL:

**Service Principal (client credentials flow):**
```yaml
auth:
  method: service_principal
  tenant_id: "${AZURE_TENANT_ID}"
  client_id: "${AZURE_CLIENT_ID}"
  client_secret: "${AZURE_CLIENT_SECRET}"
```

**Managed Identity:**
```yaml
auth:
  method: managed_identity
```

Implementation uses `azure-identity`:
- Service principal: `ClientSecretCredential(tenant_id, client_id, client_secret)`
- Managed identity: `DefaultAzureCredential()` (or `ManagedIdentityCredential()` for explicit selection)

Token scope: `https://<environment_url>/.default`

Token caching and refresh are handled by `azure-identity` internally. The client acquires a token before each request batch and reuses it within its validity window.

No interactive/device code auth — incompatible with headless automated pipelines.

**Production recommendation:** For production Azure-hosted deployments, prefer `ManagedIdentityCredential()` over `DefaultAzureCredential()` for predictable behavior. `DefaultAzureCredential` probes multiple credential providers in sequence — a provider available at `on_start()` may become unavailable mid-pagination, causing a hard failure with no resume path. `ManagedIdentityCredential()` is explicit and deterministic.

### Authentication Config Model

**Divergence from `AzureAuthConfig` (deliberate):** The existing `AzureAuthConfig` in `plugins/infrastructure/azure_auth.py` supports four auth methods (connection string, SAS token, managed identity, service principal) using boolean flags. `DataverseAuthConfig` supports only OAuth2 methods (service principal, managed identity) using a `Literal` discriminator. This divergence is intentional — Dataverse is a REST API secured by Azure Entra ID, not a storage service. Connection strings and SAS tokens are Azure Storage concepts with no Dataverse equivalent. If a future OAuth2-only Azure plugin is added, consider extracting a shared `AzureOAuth2AuthConfig` base from both configs.

```python
class DataverseAuthConfig(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    method: Literal["service_principal", "managed_identity"]

    # Service principal fields (required when method=service_principal)
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

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
```

### OData Query Execution

Two query paths:

**Structured OData queries:**
```
GET /api/data/v9.2/<entity>?$select=field1,field2&$filter=statecode eq 0&$orderby=createdon desc&$top=5000
```

**FetchXML queries:**
```
GET /api/data/v9.2/<entity>?fetchXml=<url-encoded-xml>
```

Both return the same JSON response shape:
```json
{
  "value": [{"field1": "...", "field2": "..."}, ...],
  "@odata.nextLink": "https://...?$skiptoken=..."
}
```

### Pagination

Two pagination mechanisms, depending on query mode:

**Structured OData queries:** Server-driven pagination via `@odata.nextLink`. The client follows next-link URLs, yielding pages to the caller. **SSRF guard: every `@odata.nextLink` URL MUST pass two-layer validation before following.** The `@odata.nextLink` value is Tier 3 data (returned from an external API response). A malicious or compromised endpoint could return a nextLink pointing to an internal host (e.g., `http://169.254.169.254/metadata/identity/...`). Validation is two-layered:

1. **Domain allowlist pre-filter** — the nextLink hostname must match the same allowlist patterns used for `environment_url` validation (see Security Considerations). This rejects obviously wrong domains cheaply before DNS resolution.
2. **IP-pinning validation** — the nextLink URL is then passed through `validate_url_for_ssrf()` from `core/security/web.py`, which resolves the hostname to an IP and validates it against `ALWAYS_BLOCKED_RANGES` (cloud metadata endpoints, private ranges). This prevents DNS rebinding attacks where a hostname matching `*.crm.dynamics.com` resolves to an internal IP at request time.

Both layers must pass. Reject and fail the pagination if either check fails. The domain allowlist alone (via `fnmatch`) is insufficient — it validates the hostname string but not the IP it resolves to. The `validate_url_for_ssrf()` infrastructure already exists and handles the IP-pinning concern for `AuditedHTTPClient.get_ssrf_safe()`.

**Audit recording on SSRF rejection (MANDATORY):** When SSRF validation rejects a `@odata.nextLink` URL, the source plugin MUST record the rejection via `ctx.record_call()` with `CallStatus.ERROR` **before** raising the non-retryable exception. Per CLAUDE.md: "if it's not recorded, it didn't happen." Without this record, an auditor examining a failed run would see the last successful page call and then a run failure, with no intermediate record explaining why pagination stopped. The error payload should include the rejected URL hostname and the validation layer that rejected it (domain allowlist or IP-pinning). The same audit recording applies to the **empty-page guard** termination — record a `CallStatus.ERROR` entry with the consecutive empty page count before raising.

**Empty-page guard:** If a page returns an empty `value` array alongside an `@odata.nextLink`, the client increments a consecutive empty page counter. After 3 consecutive empty pages, pagination terminates with a descriptive error classified as non-retryable. This prevents runaway infinite loops from Dataverse edge conditions where the server returns a nextLink but no data. The counter resets to zero whenever a page returns at least one row.

**FetchXML queries:** Paging cookie mechanism. Dataverse returns a `@Microsoft.Dynamics.CRM.fetchxmlpagingcookie` in the response. The client URL-decodes the cookie and injects it as a `paging-cookie` attribute on the `<fetch>` element using `xml.etree.ElementTree` (or equivalent XML-aware API). **The paging cookie is Tier 3 data — it MUST NOT be injected via string formatting or concatenation.** The XML library handles attribute escaping automatically, preventing injection if the cookie contains XML metacharacters (`"`, `>`, `&`). The client also increments the `page` attribute via the ElementTree API and re-serializes with `ET.tostring()`. Pagination ends when the response contains `@Microsoft.Dynamics.CRM.morerecords: false`.

**Root element validation (MANDATORY):** Before injecting the paging cookie, the client MUST validate that the parsed XML root element is `<fetch>`. If the root element is not `<fetch>`, raise a non-retryable `DataverseClientError` with the actual root tag in the error message. Without this check, `ElementTree.set()` would silently modify whatever the root element happens to be — producing a malformed FetchXML query that Dataverse may accept but misinterpret. This is an offensive programming guard per CLAUDE.md — we know what the structure should be, so assert it:

```python
import xml.etree.ElementTree as ET

# In DataverseClient pagination loop:
#   fetch_xml: str = the current FetchXML query string (from config or previous iteration)
#   decoded_cookie: str = URL-decoded @Microsoft.Dynamics.CRM.fetchxmlpagingcookie from response
#   next_page: int = current page number + 1
root = ET.fromstring(fetch_xml)
if root.tag != "fetch":
    raise DataverseClientError(
        f"FetchXML root element must be <fetch>, got <{root.tag}>. "
        f"Paging cookie injection requires a valid FetchXML structure.",
        retryable=False,
    )
root.set("paging-cookie", decoded_cookie)
root.set("page", str(next_page))
```

Both mechanisms provide:
- Memory-efficient streaming (one page in memory at a time)
- No client-side offset tracking
- Respects Dataverse's preferred page size

**Data consistency note:** OData structured queries without `$orderby` on a stable key field may produce inconsistent page sets on long-running reads (records added or deleted between page requests). Recommend `$orderby=createdon asc` or an equivalent stable sort for reproducible page walks. FetchXML queries with paging cookies have similar eventual-consistency semantics — Dataverse does not guarantee snapshot isolation across pages.

The client yields `list[dict]` per page, not individual rows. The source plugin handles row-level iteration.

### Batch Requests (Deferred)

**`$batch` support is out of scope for the initial implementation.** The day-one sink uses individual HTTP requests per row, which is correct, auditable, and sufficient for initial throughput requirements.

When throughput data justifies it, a `$batch` mode can be added as a tracked optimization task. Dataverse supports `POST /$batch` with OData batch format (up to 1000 operations per batch, multipart/mixed content type, change sets for transactional grouping). The `$batch` implementation would need to handle partial success internally (re-submitting failed rows individually) before returning a single `ArtifactDescriptor`.

### Rate Limiting

Two layers:
1. **Hard ceiling** — configured at the settings level via `RateLimitSettings` and injected through the `RateLimitRegistry`. The `DataverseClient` accepts an optional `RateLimiter` instance (same pattern as `AuditedHTTPClient` and `AuditedLLMClient`). This is the IP-ban prevention guard — never exceeded regardless of adaptive behavior.
2. **Adaptive backoff (classification only)** — when a 429 response arrives with `Retry-After` header, the client does NOT sleep internally. Instead, it raises a retryable exception with the `Retry-After` duration attached as exception metadata. **Important: the `Retry-After` value is recorded in the exception metadata and audit trail but is NOT used for wait timing.** `RetryManager.execute_with_retry()` uses `wait_exponential_jitter` unconditionally and has no `wait_for` callback to consume server-specified wait durations. The engine's exponential backoff applies regardless of the `Retry-After` value. Adding `Retry-After`-aware wait support to `RetryManager` is tracked as a separate future enhancement. **The client MUST still clamp `Retry-After` for classification purposes:** `effective_retry_after = max(1, min(retry_after, retry_after_cap))`. The floor (1 second) normalizes `Retry-After: 0` and negative values for metadata recording. The ceiling (configurable, default: 60 seconds) serves as a classification gate — if `Retry-After` exceeds the cap, the client classifies the error as non-retryable (the server is requesting an unreasonable pause that indicates a systemic issue rather than a transient rate limit). This prevents a large `Retry-After` value (e.g., 3600 seconds) from being classified as retryable when the server is signaling extended unavailability.

```yaml
# Settings-level rate limiting (not per-plugin config)
rate_limit:
  default_requests_per_minute: 900   # Hard ceiling (15/s equivalent)
```

**Rate limiter keys:** The source and sink register distinct rate limiter keys to prevent contention within the same pipeline: `dataverse_source` and `dataverse_sink`. This ensures a Dataverse source + Dataverse sink pipeline does not compete for a single shared budget. Other plugins (LLM, HTTP transforms) use their own keys and are unaffected. Key names use underscore-separated identifiers to satisfy the `RateLimiter` name validation constraint (`^[a-zA-Z][a-zA-Z0-9_]*$`).

**Note on per-environment granularity:** The keys are plugin-scoped, not environment-scoped. If a pipeline reads from Dataverse environment A and writes to Dataverse environment B, both share the same per-plugin budget. This is consistent with the existing pattern (`web_scrape`, `openrouter`) and sufficient for day-one. If per-environment rate limiting is needed, the keys can be extended to `dataverse_source_<sanitized_host>` (dots replaced with underscores) in a future iteration.

The `DataverseClient` constructor accepts `limiter: RateLimiter | None` (obtained from `RateLimitRegistry` via the `LifecycleContext.rate_limit_registry` using the appropriate key) and acquires the rate limit before each request.

### DataverseClientError Exception Class

```python
class DataverseClientError(Exception):
    """Exception for Dataverse protocol-level errors.

    Raised by DataverseClient for HTTP errors, JSON parse failures, and
    SSRF validation rejections. The source/sink plugin catches these to
    record audit entries via ctx.record_call() before re-raising for
    engine retry/quarantine handling.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.latency_ms = latency_ms
```

This exception lives in `plugins/infrastructure/clients/dataverse.py` alongside `DataverseClient`. It does NOT inherit from `PluginRetryableError` — `DataverseClient` is used by sources and sinks, not transforms. The engine's `_execute_transform_with_retry` never sees this exception directly; the source/sink plugin catches it, records the audit entry, and re-raises for the engine's source/sink error handling.

### Error Classification

| HTTP Status | Classification | Engine Action |
|-------------|---------------|---------------|
| 200-299 | Success | Process response |
| 400 | Non-retryable | Config/data error — quarantine row (sink) or fail run (source) |
| 401 | Retryable (once) | Auth failure — discard the current credential instance and construct a fresh one (e.g., new `ClientSecretCredential` or `ManagedIdentityCredential`), then retry once. `azure-identity`'s `get_token()` does not expose a `force_refresh` parameter — the only reliable way to force re-authentication is credential reconstruction. If the retry also returns 401, classify as non-retryable and fail run. This recovers from transient token expiry without masking genuine auth misconfigurations. |
| 403 | Non-retryable | Authorization failure — fail run |
| 404 | Non-retryable | Entity/record not found — quarantine row (sink) or fail run (source) |
| 409 | Non-retryable | Conflict (duplicate on create) — quarantine row |
| 412 | Non-retryable | Precondition failed (optimistic concurrency) — quarantine row |
| 429 | Retryable | Rate limited — `Retry-After` recorded in metadata but engine uses exponential backoff (see Rate Limiting) |
| 500-599 | Retryable | Transient server error — retry |
| Network error | Retryable | Timeout, connection reset — retry |

### Tier 3 Boundary

Every response from Dataverse is Tier 3 (external data). The client validates immediately:
- JSON parse (reject malformed responses)
- **NaN/Infinity rejection** — response JSON is parsed with strict non-finite value detection, using the shared `parse_json_strict()` / `contains_non_finite()` utilities extracted from `AuditedHTTPClient` (see `plugins/infrastructure/clients/json_utils.py`). NaN and Infinity values in Dataverse responses would survive Python's `json.loads()` but crash in `rfc8785` canonicalization during audit recording. Detecting them here — at the Tier 3 boundary — produces a clean Dataverse-attributed error rather than a cryptic canonicalization crash two layers downstream

**CRITICAL: `parse_json_strict()` returns a tuple, NOT an exception.** The function signature is `parse_json_strict(text) -> tuple[Any, str | None]` — it returns `(parsed_value, None)` on success or `(None, error_message)` on failure. This is the existing `http.py` pattern where callers check the error and return a sentinel dict. `DataverseClient` MUST explicitly check the error return and raise, not silently propagate a `None` value:

```python
from elspeth.plugins.infrastructure.clients.json_utils import parse_json_strict

parsed, error = parse_json_strict(response.text)
if error is not None:
    raise DataverseClientError(
        f"Invalid JSON from Dataverse: {error}",
        retryable=False,
        status_code=response.status_code,
    )
# parsed is now Tier 2 — validated
```
- Response structure (`value` array present for queries, error object for failures)
- Type validation of OData metadata fields (`@odata.nextLink` is string or absent)

Once validated, the parsed data is Tier 2 — trusted by the source/sink plugins.

### Response Type

`DataverseClient` methods return a frozen dataclass representing a validated page response:

```python
@dataclass(frozen=True, slots=True)
class DataversePageResponse:
    status_code: int
    rows: list[dict[str, Any]]
    latency_ms: float
    headers: dict[str, str]
    next_link: str | None  # @odata.nextLink URL, if present
    paging_cookie: str | None  # FetchXML paging cookie, if present
    more_records: bool  # True if more pages exist

    def __post_init__(self) -> None:
        if self.next_link is not None and self.paging_cookie is not None:
            raise ValueError(
                "next_link and paging_cookie are mutually exclusive — "
                "OData queries use next_link, FetchXML queries use paging_cookie"
            )
```

This type is constructed by `DataverseClient` after Tier 3 validation (JSON parse, NaN/Infinity rejection, structure validation). The source/sink plugin consumes it as Tier 2 data. The `next_link` and `paging_cookie` fields are mutually exclusive — structured OData queries use `next_link`, FetchXML queries use `paging_cookie` + `more_records`. The `__post_init__` enforces this invariant per CLAUDE.md offensive programming principles.

## DataverseSource

### Extends BaseSource

```python
class DataverseSource(BaseSource):
    name = "dataverse"
    determinism = Determinism.EXTERNAL_CALL  # Live REST API, not static file read
```

### Configuration

```python
class DataverseSourceConfig(DataPluginConfig):
    environment_url: str              # https://<org>.crm.dynamics.com
    auth: DataverseAuthConfig
    api_version: str = "v9.2"         # Dataverse Web API version
    on_validation_failure: str                 # Route for invalid rows (required — no default, matching AzureBlobSourceConfig pattern)

    # Structured query mode
    entity: str | None = None         # Entity logical name (e.g., "contact")
    select: list[str] | None = None   # $select fields (None = all)
    filter: str | None = None         # $filter expression (static OData only — see note below)
    orderby: str | None = None        # $orderby expression
    top: int | None = None            # $top limit (None = all records)

    # FetchXML query mode
    fetch_xml: str | None = None      # Raw FetchXML string

    # Field handling (normalization is mandatory — logical names are always normalized)
    field_mapping: dict[str, str] | None = None  # Manual field name overrides
    include_formatted_values: bool = False  # Preserve Dataverse formatted value annotations

    @field_validator("environment_url")
    @classmethod
    def validate_environment_url_https(cls, v: str) -> str:
        """HTTPS required. Bearer tokens sent over plain HTTP would be unencrypted."""
        import urllib.parse
        parsed = urllib.parse.urlparse(v)
        if parsed.scheme != "https":
            raise ValueError(
                f"environment_url must use HTTPS scheme, got {parsed.scheme!r}. "
                f"Bearer tokens are sent in Authorization headers — HTTP would expose them in transit."
            )
        return v

    @model_validator(mode="after")
    def validate_query_mode(self) -> Self:
        has_structured = self.entity is not None
        has_fetchxml = self.fetch_xml is not None
        if has_structured == has_fetchxml:
            raise ValueError("Specify exactly one of: entity (structured query) or fetch_xml")
        if not has_structured and any(f is not None for f in (self.select, self.filter, self.orderby, self.top)):
            raise ValueError("select/filter/orderby/top require entity (structured query mode)")
        return self
```

**`$filter` is static only:** The `filter` field is static OData configuration authored by the pipeline operator. Row-value interpolation into `$filter` is explicitly out of scope — adding it would require OData-safe escaping to prevent injection.

Note: Extends `DataPluginConfig` directly (not `SourceDataConfig`/`PathConfig`, which require a local file `path`). Follows the `AzureBlobSourceConfig` pattern for REST API sources. The `on_validation_failure` field is defined directly for quarantine routing. Rate limiting is NOT a per-plugin config field — it is configured at the settings level via `RateLimitSettings` and injected through the `RateLimitRegistry` at runtime (see Rate Limiting section).

### Load Lifecycle

```
__init__(config)
  → Validate config (Pydantic)
  → Store auth config and query parameters
  → Pre-compile FetchXML (if provided) — structural validation only
  → NOTE: DataverseClient is NOT constructed here (needs landscape/run_id from context)

on_start(ctx: LifecycleContext)
  → Store ctx.run_id, ctx.telemetry_emit, ctx.rate_limit_registry
  → Construct credential (azure-identity) — validates credentials early
  → Obtain rate limiter (with null guard):
    limiter = ctx.rate_limit_registry.get_limiter("dataverse_source") if ctx.rate_limit_registry is not None else None
    NOTE: LifecycleContext.rate_limit_registry is typed RateLimitRegistry | None.
    When rate limiting is disabled, the registry is None. Follow the existing pattern
    in azure/base.py:132 and llm/transform.py:1068.
  → Construct DataverseClient with credential, rate_limiter (may be None), httpx.Client
  → NOTE: DataverseClient is a pure protocol client — it handles auth, pagination,
    rate limiting, and error classification, but does NOT record audit calls.
    The source plugin calls ctx.record_call() after each HTTP call.
  → Optionally validate entity exists via metadata query (response recorded via ctx.record_call)

load(ctx: SourceContext)
  → Execute query (structured or FetchXML)
  → Follow pagination:
    → Structured queries: @odata.nextLink URLs (each validated against SSRF allowlist)
    → FetchXML queries: paging cookie via xml.etree.ElementTree (XML-safe injection)
  → Per page:
    → Call DataverseClient.get_page() (returns response data)
    → Record HTTP call via ctx.record_call() (audit trail entry per page fetch)
    → Per row:
      → Strip OData metadata (@odata.* fields)
      → Optionally preserve formatted values (include_formatted_values)
      → Normalize field names (if enabled)
      → Validate against schema
      → yield SourceRow.valid(row) or SourceRow.quarantined(row, error, destination)
  → Lock schema contract after first valid row
  → If zero valid rows yielded across ALL pages: force-lock contract with empty schema.
    NOTE: Unlike AzureBlobSource (single file, no pagination), DataverseSource defers force-lock
    until source exhaustion — not first-page completion. If page 1 produces zero valid rows but
    more pages exist, the schema remains unlocked and subsequent pages can still establish the
    contract. Only after all pages are consumed with zero valid rows across the entire result set
    does the empty-schema force-lock apply. This prevents a transient quarantine spike on page 1
    from silently discarding valid data on pages 2+.

    IMPLEMENTATION NOTE — schema lock flag scoping: The `first_valid_row_processed` flag
    that controls schema lock timing MUST be an instance-level flag (e.g., `self._schema_locked`)
    initialized to False before the pagination loop begins. It is set to True on the first valid
    row across ALL pages — it is NOT reset per page. If an implementor places this flag inside
    the per-page loop body and resets it at each page start, observed-mode schema inference
    fires on every page, causing type mismatch quarantines on rows in pages 2+ if the inferred
    types differ even slightly from page 1.

on_complete(ctx: LifecycleContext)
  → Emit source statistics via ctx.telemetry_emit (rows yielded, pages fetched, quarantine count)
  → NOTE: Do NOT use logger for pipeline statistics — per CLAUDE.md logging policy,
    operational statistics go through telemetry, not logging.

close()
  → Release DataverseClient resources (HTTP connections, credential handles)
```

**Auth token lifecycle:** The credential object (`ClientSecretCredential` or `DefaultAzureCredential`) is constructed once in `on_start()`. `azure-identity` handles token caching and transparent refresh internally — `credential.get_token(scopes)` is called before each page request (not once at pagination start), so long pagination sequences spanning token validity windows are handled automatically. If a transient network failure prevents background token refresh, the next request gets a stale token and a 401 — the client force-refreshes once before classifying as non-retryable (see Error Classification).

**Audit recording pattern:** Sources and sinks do not have access to `state_id` (which is a per-row concept on `TransformContext`). Instead, they use `SourceContext.record_call()` / `SinkContext.record_call()` to record external API calls in the Landscape `calls` table. The context method signature is `record_call(call_type, status, request_data, response_data, latency_ms, *, provider)` — the context handles `state_id` and `call_index` allocation internally. This is the same pattern used by `AzureBlobSource` at `azure_blob_source.py:434`. The `DataverseClient` does NOT hold a `LandscapeRecorder` reference — it returns response data to the plugin, which calls `ctx.record_call()`.

**Non-resumability:** Dataverse source pipelines are non-resumable end-to-end. The source does not checkpoint page progress, and the sink declares `supports_resume = False` (Dataverse writes are not locally staged). If the pipeline crashes after processing N pages, `elspeth resume` is rejected at the CLI. The only recovery is a full re-run from page 1. For large datasets, pair with **upsert mode** to make re-runs idempotent. Operators should be aware of this cost when designing pipelines against large Dataverse entities.

### OData Metadata Stripping

Dataverse responses include OData annotations (`@odata.etag`, `@odata.context`, `_field_value@OData.Community.Display.V1.FormattedValue`, etc.). The source strips these from row data before yielding. The raw responses are available in the audit trail (recorded via `record_call()`) but OData annotations don't pollute the pipeline row.

Option: `include_formatted_values: true` to preserve Dataverse's formatted value annotations as additional fields (e.g., `statecode` = 0, `statecode__formatted` = "Active"). Off by default. **Collision risk:** The `__formatted` suffix is applied to the normalized field name. If the source entity has both `statecode` and a field that normalizes to `statecode__formatted`, the names collide. The source MUST detect this collision during schema discovery and fail with a descriptive error rather than silently overwriting.

### Schema Modes

All three schema modes (fixed, flexible, observed) are supported:
- **Observed** (default): Discover fields from first page response, lock after first valid row. **Known limitation:** In observed mode, the schema is locked after the first valid row. Fields appearing on later pages that were absent on the first page are silently dropped. This is a known limitation of observed mode with eventual-consistency APIs — Dataverse does not guarantee field-set consistency across pages
- **Fixed**: User declares expected fields; extras rejected
- **Flexible**: User declares known fields; extras discovered from first page

### Config Examples

**Structured query:**
```yaml
source:
  plugin: dataverse
  options:
    environment_url: "https://myorg.crm.dynamics.com"
    auth:
      method: service_principal
      tenant_id: "${AZURE_TENANT_ID}"
      client_id: "${AZURE_CLIENT_ID}"
      client_secret: "${AZURE_CLIENT_SECRET}"
    entity: "contact"
    select: [fullname, emailaddress1, createdon, _parentcustomerid_value]
    filter: "statecode eq 0 and createdon gt 2024-01-01"
    orderby: "createdon desc"
    top: 5000
    schema:
      mode: observed
```

**FetchXML query with joins:**
```yaml
source:
  plugin: dataverse
  options:
    environment_url: "https://myorg.crm.dynamics.com"
    auth:
      method: managed_identity
    fetch_xml: |
      <fetch top="5000">
        <entity name="contact">
          <attribute name="fullname"/>
          <attribute name="emailaddress1"/>
          <link-entity name="account" from="accountid" to="parentcustomerid">
            <attribute name="name" alias="company_name"/>
          </link-entity>
        </entity>
      </fetch>
    schema:
      mode: observed
```

## DataverseSink

### Extends BaseSink

```python
class DataverseSink(BaseSink):
    name = "dataverse"
    idempotent = False  # Only upsert mode is idempotent; create/update are not
    supports_resume = False  # Dataverse writes are not locally staged
```

Note: `idempotent` is False at the class level. In upsert mode the *operation* is idempotent (PATCH is naturally idempotent), but the sink class conservatively declares False since create and update modes are not.

### Configuration

```python
class DataverseSinkConfig(DataPluginConfig):
    environment_url: str
    auth: DataverseAuthConfig
    api_version: str = "v9.2"

    entity: str                       # Target entity logical name
    mode: Literal["upsert"] = "upsert"  # Day-one: upsert only (see "Deferred Write Modes")

    @field_validator("environment_url")
    @classmethod
    def validate_environment_url_https(cls, v: str) -> str:
        """HTTPS required — same validator as DataverseSourceConfig."""
        import urllib.parse
        parsed = urllib.parse.urlparse(v)
        if parsed.scheme != "https":
            raise ValueError(
                f"environment_url must use HTTPS scheme, got {parsed.scheme!r}. "
                f"Bearer tokens are sent in Authorization headers — HTTP would expose them in transit."
            )
        return v

    # Field mapping (mandatory — no passthrough)
    field_mapping: dict[str, str]     # pipeline_field → dataverse_column

    # Key field (required for upsert)
    alternate_key: str                # Business key field for upsert (PATCH with alternate key)

    # Lookup field declarations
    lookups: dict[str, LookupConfig] | None = None
```

Note: `DataPluginConfig` provides `schema_config` for input validation. Rate limiting uses the settings-level `RateLimitRegistry`, not a per-plugin config field.

### Lookup Configuration

```python
class LookupConfig(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    target_entity: str    # Dataverse entity to bind to (e.g., "accounts")
    target_field: str     # Navigation property name (e.g., "parentcustomerid")
```

The sink translates lookup fields into OData bind syntax:
```python
# Pipeline row: {"parent_account_id": "abc-123-guid"}
# Config: lookups.parent_account_id = {target_entity: "accounts", target_field: "parentcustomerid"}
# OData payload: {"parentcustomerid@odata.bind": "/accounts(abc-123-guid)"}
```

### Write Mode: Upsert

Day-one implementation supports **upsert only**. PATCH is naturally idempotent — safe for retryable pipelines and crash recovery re-runs.

| Mode | HTTP Method | URL Pattern | Error on |
|------|-------------|-------------|----------|
| `upsert` | PATCH | `/api/data/v9.2/<entity>(<alternate_key>=<url_encoded_value>)` | — (creates or updates) |

The `alternate_key` value is URL-encoded before interpolation into the URL path to prevent injection via special characters (`/`, `(`, `)`, `=`). **Validation:** Before URL construction, the `alternate_key` value from the pipeline row is validated — empty string, `None`, or whitespace-only values produce a `TransformResult.error()` with `retryable=False` rather than a malformed PATCH URL. This is a row-level data validation (Tier 2 operation wrapping), not a config validation — the config field `alternate_key` names the key field, but the value comes from each row.

### Deferred Write Modes

**`create` and `update` modes are deferred from the initial implementation.** `create` mode (POST) has an unresolvable audit integrity gap: if a POST succeeds but the response is lost (network error), the engine retries, gets a 409 (conflict), and records the row as QUARANTINED — but the data was written successfully to Dataverse. The audit trail then contains a factually incorrect record: QUARANTINED for data that exists. This violates ELSPETH's core principle: "I don't know what happened is never an acceptable answer." Upsert avoids this because PATCH is idempotent — a retry produces the same outcome.

`create` mode can be added in a future iteration with either (a) a deterministic idempotency key mechanism (e.g., `MSCRM.SuppressDuplicateDetection` header with a row-derived key), or (b) a mandatory `max_attempts: 1` constraint with documented audit implications. `update` mode (PATCH by primary key) can be added straightforwardly as it shares the idempotent PATCH semantics.

### Sink Lifecycle

```
__init__(config)
  → Validate config (Pydantic)
  → Store field mapping, lookup config, write mode
  → NOTE: DataverseClient is NOT constructed here (needs landscape/run_id from context)

on_start(ctx: LifecycleContext)
  → Store ctx.run_id, ctx.telemetry_emit, ctx.rate_limit_registry
  → Construct credential (azure-identity) — validates credentials early
  → Obtain rate limiter (with null guard):
    limiter = ctx.rate_limit_registry.get_limiter("dataverse_sink") if ctx.rate_limit_registry is not None else None
    NOTE: Same null guard as source — LifecycleContext.rate_limit_registry is RateLimitRegistry | None.
  → Construct DataverseClient with credential, rate_limiter (may be None), httpx.Client
  → NOTE: DataverseClient is a pure protocol client — it handles auth, rate limiting,
    and error classification, but does NOT record audit calls.
    The sink plugin calls ctx.record_call() after each HTTP call.
  → Optionally validate entity/alternate key exists via metadata query

write(rows, ctx: SinkContext)
  → Process rows serially (one at a time):
    → Apply field_mapping (pipeline names → Dataverse column names)
    → URL-encode alternate_key value for PATCH URL construction
    → Format lookup fields (@odata.bind syntax)
    → Validate all mapped fields present (crash if missing — Tier 2 data)
    → Submit PATCH request to DataverseClient
    → Record via ctx.record_call() (one audit trail entry per row write)
    → On row-level error: raise typed exception immediately (engine handles retry/quarantine)
  → If all rows succeed: return ArtifactDescriptor with batch metadata.
    NOTE: ArtifactDescriptor has factory methods for_file(), for_database(), for_webhook() —
    none maps cleanly to "batch of REST API upserts". Options: (a) use for_webhook() with
    adapted semantics (closest fit — external HTTP endpoint), (b) add a for_api() factory to
    contracts/results.py, or (c) construct ArtifactDescriptor directly. Decide during
    implementation — this is not blocking.
  → NOTE: On failure, `SinkExecutor` calls `sink.write(rows, ctx)` with the full batch.
    On exception, all states in the batch are marked FAILED, and the engine retries the
    entire batch — not just the failed row. Previously-written rows are re-sent. PATCH
    idempotency makes this safe for upsert mode: re-sending a successful PATCH produces
    the same outcome.

flush()
  → No-op (Dataverse writes are immediate, no local staging buffer)

close()
  → Release DataverseClient resources (HTTP connections, credential handles)
```

### Write Strategy: Individual Requests

The `write()` method receives a list of rows from the engine. The sink processes rows **serially** — one HTTP request per row, in order. On success, a single `ArtifactDescriptor` is returned for the batch. On failure, the sink raises an exception on the **first** row that fails. `SinkExecutor` calls `sink.write(rows, ctx)` with the full batch — on exception, all states in the batch are marked FAILED, and the engine retries the **entire batch**, not just the failed row. Previously-written rows are re-sent. PATCH idempotency makes this safe for upsert mode: re-sending a successful PATCH produces the same outcome, so no duplicate writes or data corruption occurs.

For throughput, the `DataverseClient` uses `httpx`'s connection pooling and keep-alive, and the settings-level rate limiter prevents overloading the API.

Each write request is recorded individually via `SinkContext.record_call()` — the audit trail contains one `calls` table entry per row write, providing full per-row lineage. **Note:** `SinkContext.record_call()` operates with `state_id=None` during `write()` (the `SinkExecutor` clears it before calling the sink). The context routes audit recording through `operation_id` instead. This is the same pattern used by `AzureBlobSource`'s load-phase recording.

The `DataverseClient` issues PATCH requests directly via `httpx.Client`. The PATCH method is naturally idempotent, making upsert safe for retryable pipelines and crash recovery re-runs.

**`$batch` is explicitly out of scope for the initial implementation.** OData batch format (multipart/mixed with Content-ID references and change set scoping) is non-trivial to implement correctly, and has no consumer in the day-one design. The individual-request path is correct, auditable, and sufficient. A `$batch` mode can be added as a tracked optimization task when throughput data justifies it.

### Config Example

```yaml
sinks:
  dataverse_output:
    plugin: dataverse
    options:
      environment_url: "https://myorg.crm.dynamics.com"
      auth:
        method: service_principal
        tenant_id: "${AZURE_TENANT_ID}"
        client_id: "${AZURE_CLIENT_ID}"
        client_secret: "${AZURE_CLIENT_SECRET}"
      entity: "contact"
      mode: upsert
      alternate_key: emailaddress1
      field_mapping:
        customer_email: emailaddress1
        customer_name: fullname
        sentiment_score: new_sentimentscore
      lookups:
        parent_account_id:
          target_entity: accounts
          target_field: parentcustomerid
```

## Dependencies

All Dataverse dependencies are already available via the existing `[azure]` extra or base dependencies:

| Dependency | Extra | Purpose |
|-----------|-------|---------|
| `httpx` | base | HTTP client (already used by `AuditedHTTPClient`) |
| `azure-identity` | `[azure]` | OAuth2 token acquisition |

No new dependencies required. The Dataverse Web API is a standard OData v4 REST endpoint — no dedicated SDK needed.

## Plugin Discovery

Both plugins must be discoverable by the plugin scanner. The scanner (`plugins/infrastructure/discovery.py`) uses `PLUGIN_SCAN_CONFIG` to determine which directories to scan. The source and sink directories are already scanned, so `dataverse.py` files placed there will be discovered automatically.

No changes to `PLUGIN_SCAN_CONFIG` are needed.

## Testing Strategy

### Test File Locations

```
tests/unit/plugins/infrastructure/clients/test_fingerprinting.py
tests/unit/plugins/infrastructure/clients/test_json_utils.py
tests/unit/plugins/infrastructure/clients/test_dataverse_client.py
tests/unit/plugins/sources/test_dataverse_source.py
tests/unit/plugins/sinks/test_dataverse_sink.py
tests/integration/plugins/test_dataverse_pipeline.py
```

**Run commands:**
```bash
.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/test_dataverse_client.py -v
.venv/bin/python -m pytest tests/unit/plugins/sources/test_dataverse_source.py -v
.venv/bin/python -m pytest tests/unit/plugins/sinks/test_dataverse_sink.py -v
.venv/bin/python -m pytest tests/integration/plugins/test_dataverse_pipeline.py -v
```

### Unit Tests

- `fingerprinting.py`: HMAC fingerprinting of sensitive headers (bearer tokens, subscription keys), non-sensitive header passthrough, `ELSPETH_FINGERPRINT_KEY` absence check, byte-identical output regression test against original `http.py` implementations
- `json_utils.py`: strict JSON parsing rejecting NaN/Infinity, valid JSON passthrough, edge cases (nested NaN, Infinity in arrays), byte-identical output regression test against original `http.py` implementations
- `DataverseAuthConfig` validation (service principal missing fields, managed identity, mutual exclusion)
- `DataverseSourceConfig` validation (structured vs FetchXML mutual exclusion, required fields, `on_validation_failure` required)
- `DataverseSinkConfig` validation (alternate_key required, field mapping, lookup config)
- OData metadata stripping (various annotation patterns, `include_formatted_values` collision detection)
- Field mapping and lookup binding syntax generation
- URL encoding of alternate_key values with special characters (`/`, `(`, `)`, `=`)
- Error classification (status code → retryable/non-retryable, 401 force-refresh path)
- Pagination link following (mock responses with `@odata.nextLink`)
- `@odata.nextLink` SSRF validation (reject cross-host nextLink URLs)
- FetchXML paging cookie injection via ElementTree (cookie with XML metacharacters)
- Retry-After cap enforcement (value exceeding max → non-retryable)

### Integration Tests

- Source: structured query against mock Dataverse endpoint (httpx mock transport)
- Source: FetchXML query against mock endpoint with paging cookie round-trip
- Source: pagination across multiple pages with nextLink SSRF validation
- Source: schema discovery and contract locking
- Source: auth token refresh during pagination (mock credential expiry)
- Sink: upsert mode against mock endpoint
- Sink: partial batch failure (raise on first failing row, prior rows committed)
- Sink: lookup field binding in request payload
- End-to-end: source → transform → sink pipeline with mock Dataverse

### Tier Model Compliance

All tests exercise production code paths (no manual object construction that bypasses `from_dict()` or `instantiate_plugins_from_config()`). The `DataverseClient` is tested via mock HTTP transport (httpx mock transport), not by mocking the client itself.

### Tier Model Enforcer

The three new files will trigger tier model violations that require allowlist entries in `config/cicd/enforce_tier_model/plugins.yaml`:

- `plugins/sources/dataverse.py` — covered by existing `plugins/sources/*` pattern rules
- `plugins/sinks/dataverse.py` — requires new per-file rules (R1, R2, R4, R6, R9) or a `plugins/sinks/*` wildcard pattern matching the `plugins/sources/*` pattern
- `plugins/infrastructure/clients/dataverse.py` — requires new per-file rules matching the existing client file patterns

Additionally, `plugins/infrastructure/clients/fingerprinting.py` and `plugins/infrastructure/clients/json_utils.py` may need allowlist entries in `config/cicd/enforce_tier_model/plugins.yaml` if they contain patterns that trigger tier model violations (e.g., `isinstance` checks on header values, `.get()` calls on external data dictionaries).

These allowlist entries MUST be added before the CI build will pass. Run `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` to verify.

## Security Considerations

- **Secret handling:** Client secret and API keys are resolved via ELSPETH's secret management (`secrets:` section in settings.yaml or environment variables). Never logged or stored in audit trail — HMAC fingerprints only.
- **SSRF:** Dataverse URLs are validated against an explicit allowlist of Microsoft-documented Dataverse environment domain patterns. The `environment_url` must match one of:
  - `*.crm.dynamics.com` — Commercial cloud (default)
  - `*.crm2.dynamics.com` through `*.crm11.dynamics.com` — Regional commercial instances
  - `*.crm9.dynamics.com` — US Government (GCC)
  - `*.crm.microsoftdynamics.us` — US Government (GCC High)
  - `*.crm.appsplatform.us` — US Government (DoD)
  - `*.crm.microsoftdynamics.de` — Germany (legacy)
  - `*.crm.dynamics.cn` — China (21Vianet)

  Reject any URL not matching these patterns. Do NOT use a broad `*.dynamics.com` wildcard — it would accept arbitrary subdomains on the `dynamics.com` zone. The allowlist is defined as a constant in the `DataverseClient` module.

  **Matching mechanism:** Initial domain validation uses `fnmatch.fnmatch(hostname, pattern)` against the **hostname component only** (extracted via `urllib.parse.urlparse().hostname`), not the full URL. `fnmatch` applies implicit anchoring — the pattern must match the entire hostname string, not a substring. For example, `*.crm.dynamics.com` matches `myorg.crm.dynamics.com` but does NOT match `myorg.crm.dynamics.com.attacker.example` (the trailing `.attacker.example` causes mismatch). This prevents SSRF bypass via domain suffix injection.

  **IP-pinning (defense-in-depth):** After domain allowlist validation passes, the URL is also passed through `validate_url_for_ssrf()` from `core/security/web.py`, which resolves the hostname to an IP and validates against `ALWAYS_BLOCKED_RANGES`. This prevents DNS rebinding attacks where a matching hostname resolves to a private/cloud-metadata IP. Both layers must pass.

  The allowlist can be extended via a `DATAVERSE_ADDITIONAL_DOMAINS` config setting (append-only — additions are merged with the base allowlist, not replacements). Each additional domain pattern is validated before acceptance: it must match the anchored regex `^(\*\.)?([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+\.(dynamics\.com|dynamics\.cn|microsoftdynamics\.(us|de)|appsplatform\.us)$` to ensure it targets a legitimate Microsoft sovereign cloud TLD with valid hostname labels (no dots or wildcards in label positions). Patterns that do not match this guard are rejected at config load time with a descriptive error. This requires deployment-level config access, not per-pipeline YAML.
- **`@odata.nextLink` SSRF:** Every `@odata.nextLink` URL returned in pagination responses is Tier 3 data and MUST pass two-layer validation (domain allowlist + `validate_url_for_ssrf()` IP-pinning) before the client follows it. This prevents both hostname-based and DNS-rebinding-based SSRF attacks. See the Pagination section for the full two-layer validation description.
- **FetchXML paging cookie injection:** The paging cookie returned by Dataverse is Tier 3 data. It is URL-decoded and injected into the `<fetch>` XML element using `xml.etree.ElementTree` attribute setters — **never string formatting**. The XML library handles escaping of metacharacters automatically.
- **FetchXML XXE protection:** The user-provided `fetch_xml` config string and the Dataverse paging cookie are both parsed by `xml.etree.ElementTree`. **Decision: rely on CPython's built-in XXE protection.** CPython 3.8+ disables external entity resolution in `xml.etree.ElementTree` by default — the `XMLParser` does not resolve external entities or DTDs. ELSPETH requires Python 3.12+ (see `pyproject.toml`), so this protection is guaranteed. `defusedxml` is not required but MAY be used as defense-in-depth if the dependency is already present. This decision is recorded here rather than left implicit because `fetch_xml` is user-provided config (Tier 2) parsed by an XML library with a historical XXE reputation.
- **FetchXML structural validation:** Malformed `fetch_xml` config (invalid XML syntax) is detected in `__init__` via `ET.fromstring(fetch_xml)`. A `ParseError` at this point is a **structural error** — analogous to `TemplateSyntaxError` for Jinja2 templates. It fails the run at setup, not per-row. A broken `fetch_xml` can never produce valid results for any row; deferring the error would misclassify a config problem as a data problem.
- **Token scope:** OAuth2 tokens are scoped to the specific Dataverse environment URL, not broad Azure scopes.
- **Batch atomicity:** No change sets by default — partial success is visible in the audit trail, not hidden by rollback.
- **Audit trail PII:** Row-level write recording via `ctx.record_call()` records request metadata (method, URL, status, latency) but NOT the full request body. Field names are recorded for traceability; field values are not included in `request_data` to avoid persisting PII in the audit database. The full row content is available in `node_states` via the standard Landscape tables.
