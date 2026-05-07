"""Web scraping transform with audit trail integration.

Fetches webpages, extracts content, and generates fingerprints for change detection.
Designed for compliance monitoring use cases with full audit trail integration.

Security Features:
- SSRF prevention (blocks private IPs, cloud metadata)
- URL scheme validation (HTTP/HTTPS only)
- Configurable timeouts
- Rate limiting support

Audit Trail:
- Records all HTTP calls via AuditedHTTPClient
- Stores request, raw response, and processed content in PayloadStore
- Generates fingerprints for change detection
"""

import ipaddress
from ipaddress import IPv4Network, IPv6Network
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.audit import Call
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.contract_propagation import narrow_contract_to_output
from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.core.security.web import (
    NetworkError as SSRFNetworkError,
)
from elspeth.core.security.web import (
    SSRFBlockedError,
    SSRFSafeRequest,
    validate_url_for_ssrf,
)
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.transforms.web_scrape_errors import (
    ForbiddenError,
    InvalidURLError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ServerError,
    UnauthorizedError,
    WebScrapeError,
)
from elspeth.plugins.transforms.web_scrape_extraction import extract_content
from elspeth.plugins.transforms.web_scrape_fingerprint import compute_fingerprint

if TYPE_CHECKING:
    from elspeth.contracts.plugin_assistance import PluginAssistance
    from elspeth.contracts.plugin_semantics import OutputSemanticDeclaration

# Audit-only fields — provenance metadata that lives in success_reason["metadata"],
# not in pipeline rows. See spec: 2026-03-21-audit-provenance-boundary-design.md
WEBSCRAPE_AUDIT_FIELDS: tuple[str, ...] = (
    "fetch_request_hash",
    "fetch_response_raw_hash",
    "fetch_response_processed_hash",
)


class WebScrapeHTTPConfig(BaseModel):
    """HTTP client configuration for web scrape transform.

    Controls responsible scraping behavior: abuse contact for transparency,
    scraping reason for audit trail, and timeout for resource management.
    """

    model_config = {"extra": "forbid"}

    abuse_contact: str = Field(
        ...,
        description="Email for abuse reports (required for responsible scraping)",
    )
    scraping_reason: str = Field(
        ...,
        description="Why we're scraping (recorded in audit trail)",
    )
    timeout: int = Field(
        default=30,
        gt=0,
        description="Request timeout in seconds",
    )
    allowed_hosts: str | list[str] = Field(
        default="public_only",
        description="SSRF allowlist: 'public_only' (default), 'allow_private', or list of CIDR ranges",
    )

    @field_validator("abuse_contact", "scraping_reason")
    @classmethod
    def _reject_empty(cls, v: str, info: Any) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return v

    @field_validator("allowed_hosts")
    @classmethod
    def _validate_allowed_hosts(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            if v not in ("public_only", "allow_private"):
                raise ValueError(f"allowed_hosts must be 'public_only', 'allow_private', or a list of CIDR ranges, got {v!r}")
            return v
        if not v:
            raise ValueError("allowed_hosts list must not be empty (use 'allow_private' to allow all)")
        for entry in v:
            try:
                ipaddress.ip_network(entry, strict=False)
            except ValueError as e:
                raise ValueError(f"Invalid CIDR in allowed_hosts: {entry!r}: {e}") from e
        return v


class WebScrapeConfig(TransformDataConfig):
    """Configuration for web scrape transform."""

    url_field: str
    content_field: str
    fingerprint_field: str
    format: Literal["markdown", "text", "raw"] = "markdown"
    text_separator: str = Field(
        default=" ",
        min_length=1,
        max_length=16,
        description="Separator inserted between DOM text nodes when format is text.",
    )
    fingerprint_mode: Literal["content", "full"] = "content"
    strip_elements: list[str] = Field(default_factory=lambda: ["script", "style"])
    http: WebScrapeHTTPConfig

    @field_validator("url_field", "content_field", "fingerprint_field")
    @classmethod
    def _reject_empty_field_names(cls, v: str, info: Any) -> str:
        if not v:
            raise ValueError(f"{info.field_name} must not be empty")
        return v

    @model_validator(mode="after")
    def _reject_field_collisions(self) -> "WebScrapeConfig":
        if self.content_field == self.fingerprint_field:
            raise ValueError(f"content_field and fingerprint_field must differ, both are '{self.content_field}'")
        return self


def _parse_allowed_ranges(entries: list[str]) -> tuple[IPv4Network | IPv6Network, ...]:
    """Parse allowed_hosts list entries into ip_network objects.

    Single IPs (no /) are expanded to /32 (IPv4) or /128 (IPv6).
    Uses strict=False so "10.0.0.1/8" is accepted as "10.0.0.0/8".
    """
    networks: list[IPv4Network | IPv6Network] = []
    for entry in entries:
        network = ipaddress.ip_network(entry, strict=False)
        networks.append(network)
    return tuple(networks)


def _web_scrape_added_output_fields(content_field: str, fingerprint_field: str) -> tuple[FieldDefinition, ...]:
    """Return the typed fields WebScrape guarantees on successful output rows."""
    return (
        FieldDefinition(name=content_field, field_type="str", required=True),
        FieldDefinition(name=fingerprint_field, field_type="str", required=True),
        FieldDefinition(name="fetch_status", field_type="int", required=True),
        FieldDefinition(name="fetch_url_final", field_type="str", required=True),
        FieldDefinition(name="fetch_url_final_ip", field_type="str", required=True),
    )


def _build_web_scrape_output_schema_config(
    schema_config: SchemaConfig,
    *,
    content_field: str,
    fingerprint_field: str,
) -> SchemaConfig:
    """Build the typed output contract for WebScrape's pass-through enrichment."""
    field_by_name: dict[str, FieldDefinition] = {}
    if schema_config.fields is not None:
        field_by_name.update((field.name, field) for field in schema_config.fields)

    added_fields = _web_scrape_added_output_fields(content_field, fingerprint_field)
    field_by_name.update((field.name, field) for field in added_fields)

    base_guaranteed = set(schema_config.guaranteed_fields or ())
    output_guaranteed = base_guaranteed | {field.name for field in added_fields}

    return SchemaConfig(
        # Observed input still has a known output minimum after enrichment.
        mode=schema_config.mode if schema_config.fields is not None else "flexible",
        fields=tuple(field_by_name.values()),
        guaranteed_fields=tuple(sorted(output_guaranteed)),
        audit_fields=schema_config.audit_fields,
        required_fields=schema_config.required_fields,
    )


def _build_web_scrape_output_semantics(
    *,
    content_field: str,
    format: str,
    text_separator: str,
) -> "OutputSemanticDeclaration":
    """Map WebScrapeConfig values to declared output facts for the content field."""
    from elspeth.contracts.plugin_semantics import (
        ContentKind,
        FieldSemanticFacts,
        OutputSemanticDeclaration,
        TextFraming,
    )

    if format == "markdown":
        kind = ContentKind.MARKDOWN
        framing = TextFraming.LINE_COMPATIBLE
        fact_code = "web_scrape.content.markdown"
    elif format == "raw":
        kind = ContentKind.HTML_RAW
        framing = TextFraming.NOT_TEXT
        fact_code = "web_scrape.content.raw_html"
    elif format == "text":
        kind = ContentKind.PLAIN_TEXT
        if "\n" in text_separator:
            framing = TextFraming.NEWLINE_FRAMED
            fact_code = "web_scrape.content.newline_framed_text"
        else:
            framing = TextFraming.COMPACT
            fact_code = "web_scrape.content.compact_text"
    else:
        # Unknown format value — let the schema layer handle it.
        # Returning UNKNOWN here is honest: we don't know.
        kind = ContentKind.UNKNOWN
        framing = TextFraming.UNKNOWN
        fact_code = "web_scrape.content.unknown_format"

    return OutputSemanticDeclaration(
        fields=(
            FieldSemanticFacts(
                field_name=content_field,
                content_kind=kind,
                text_framing=framing,
                fact_code=fact_code,
                configured_by=("format", "text_separator"),
            ),
        ),
    )


def _final_response_ip(response: httpx.Response) -> str:
    """Extract the final IP-pinned destination from an SSRF-safe response."""
    try:
        final_host = response.request.url.host
    except RuntimeError as e:
        raise FrameworkBugError("SSRF-safe HTTP response has no request; cannot record final resolved IP.") from e

    if final_host is None:
        raise FrameworkBugError("SSRF-safe HTTP response request URL has no host; cannot record final resolved IP.")

    try:
        ipaddress.ip_address(final_host)
    except ValueError as e:
        raise FrameworkBugError(
            f"SSRF-safe HTTP response request host {final_host!r} is not an IP address; "
            "AuditedHTTPClient must return the IP-pinned final request."
        ) from e

    return final_host


class WebScrapeTransform(BaseTransform):
    """Fetch webpages, extract content, generate fingerprints.

    Designed for compliance monitoring use cases. Features:
    - Security: SSRF prevention, URL validation
    - Audit: Full request/response recording
    - Extraction: HTML → Markdown, Text, or Raw
    - Fingerprinting: Change detection with normalization

    Configuration:
        url_field: Field containing URL to fetch
        content_field: Field to store extracted content
        fingerprint_field: Field to store content fingerprint
        format: Output format ("markdown", "text", "raw")
        text_separator: Separator between DOM text nodes when format is text
        fingerprint_mode: Fingerprinting mode ("content", "full")
        strip_elements: HTML tags to remove (default: ["script", "style"])
        http:
            abuse_contact: Email for abuse reports (required)
            scraping_reason: Why we're scraping (required)
            timeout: Request timeout in seconds (default: 30)

    Error Handling (follows LLM plugin pattern):
        - Retryable errors (5xx, 429, network): Re-raised for engine retry
        - Non-retryable errors (4xx, SSRF): Return TransformResult.error()

    Example:
        transforms:
          - plugin: web_scrape
            options:
              schema: {mode: observed}
              url_field: url
              content_field: page_content
              fingerprint_field: page_fingerprint
              format: markdown
              text_separator: "\n"  # only used with format: text
              http:
                abuse_contact: compliance@example.com
                scraping_reason: Regulatory monitoring
    """

    name = "web_scrape"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:b1339397350c2ffc"
    config_model = WebScrapeConfig
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 forward invariant."""
        return {
            "schema": {"mode": "observed"},
            "url_field": "web_scrape_probe_url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "invariants@example.com",
                "scraping_reason": "ADR-009 invariant probe",
                "allowed_hosts": ["93.184.216.34/32"],
            },
        }

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)

        # Parse and validate config
        cfg = WebScrapeConfig.from_dict(options, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)

        # Required fields
        self._url_field = cfg.url_field
        self._content_field = cfg.content_field
        self._fingerprint_field = cfg.fingerprint_field

        # Declare output fields for centralized collision detection in TransformExecutor.
        self.declared_output_fields = frozenset(
            [
                cfg.content_field,
                cfg.fingerprint_field,
                "fetch_status",
                "fetch_url_final",
                "fetch_url_final_ip",
            ]
        )

        # Format and fingerprint mode
        self._format = cfg.format
        self._text_separator = cfg.text_separator
        self._fingerprint_mode = cfg.fingerprint_mode

        # HTTP config — validated by WebScrapeHTTPConfig sub-model
        self._abuse_contact = cfg.http.abuse_contact
        self._scraping_reason = cfg.http.scraping_reason
        self._timeout = cfg.http.timeout

        # Compute allowed_ranges from allowed_hosts config
        allowed_hosts = cfg.http.allowed_hosts
        if allowed_hosts == "public_only":
            self._allowed_ranges: tuple[IPv4Network | IPv6Network, ...] = ()
        elif allowed_hosts == "allow_private":
            self._allowed_ranges = (
                ipaddress.ip_network("0.0.0.0/0"),
                ipaddress.ip_network("::/0"),
            )
        else:
            match allowed_hosts:
                case list() as cidr_ranges:
                    self._allowed_ranges = _parse_allowed_ranges(cidr_ranges)
                case _:
                    raise FrameworkBugError(
                        f"WebScrapeConfig.http.allowed_hosts validator admitted an invalid runtime value: {type(allowed_hosts).__name__!r}."
                    )

        # Element stripping
        self._strip_elements = cfg.strip_elements

        # Schema
        if cfg.schema_config is None:
            raise RuntimeError("WebScrapeTransform requires schema_config")
        self.input_schema = create_schema_from_config(
            cfg.schema_config,
            "WebScrapeInput",
            allow_coercion=False,
        )
        self._output_schema_config = _build_web_scrape_output_schema_config(
            cfg.schema_config,
            content_field=cfg.content_field,
            fingerprint_field=cfg.fingerprint_field,
        )
        self.output_schema = create_schema_from_config(
            self._output_schema_config,
            "WebScrapeOutput",
            allow_coercion=False,
        )

    def output_semantics(self) -> "OutputSemanticDeclaration":
        return _build_web_scrape_output_semantics(
            content_field=self._content_field,
            format=self._format,
            text_separator=self._text_separator,
        )

    @classmethod
    def get_agent_assistance(
        cls,
        *,
        issue_code: str | None = None,
    ) -> "PluginAssistance | None":
        from elspeth.contracts.plugin_assistance import (
            PluginAssistance,
            PluginAssistanceExample,
        )

        if issue_code != "web_scrape.content.compact_text":
            return None
        return PluginAssistance(
            plugin_name="web_scrape",
            issue_code="web_scrape.content.compact_text",
            summary=(
                "format='text' with a non-newline text_separator produces a "
                "compact single-line string. Downstream line-oriented "
                "transforms (line_explode) cannot recover line boundaries."
            ),
            suggested_fixes=(
                "Set text_separator: '\\n' to preserve line boundaries.",
                "Or use format: markdown — markdown extraction preserves line-oriented structure.",
            ),
            examples=(
                PluginAssistanceExample(
                    title="Use newline separator with text format",
                    before={"format": "text", "text_separator": " "},
                    after={"format": "text", "text_separator": "\n"},
                ),
                PluginAssistanceExample(
                    title="Switch to markdown format",
                    before={"format": "text", "text_separator": " "},
                    after={"format": "markdown"},
                ),
            ),
        )

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Inject a deterministic public-IP URL for invariant probing."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._url_field,
                value="https://93.184.216.34/invariant-probe",
            )
        ]

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: TransformContext,
    ) -> TransformResult:
        """Drive the real process path with a hermetic no-network fetch seam."""

        class _InvariantPayloadStore:
            def store(self, payload: bytes) -> str:
                return "probe-processed-hash"

        class _InvariantCall:
            request_ref = "probe-request-hash"
            response_ref = "probe-response-hash"

        def _fake_fetch_url(
            safe_request: SSRFSafeRequest,
            probe_ctx: TransformContext,
        ) -> tuple[httpx.Response, str, _InvariantCall]:
            del probe_ctx
            return (
                httpx.Response(
                    200,
                    text="<html><body><h1>Probe</h1><p>safe</p></body></html>",
                    request=httpx.Request("GET", safe_request.connection_url),
                ),
                safe_request.original_url,
                _InvariantCall(),
            )

        had_payload_store = "_payload_store" in self.__dict__
        original_payload_store: Any = None
        if had_payload_store:
            original_payload_store = self.__dict__["_payload_store"]
        had_fetch_override = "_fetch_url" in self.__dict__
        original_fetch = self._fetch_url
        try:
            self.__dict__["_payload_store"] = _InvariantPayloadStore()
            self.__dict__["_fetch_url"] = _fake_fetch_url
            return super().execute_forward_invariant_probe(probe_rows, ctx)
        finally:
            if had_payload_store:
                self.__dict__["_payload_store"] = original_payload_store
            else:
                delattr(self, "_payload_store")
            if had_fetch_override:
                self.__dict__["_fetch_url"] = original_fetch
            else:
                delattr(self, "_fetch_url")

    def on_start(self, ctx: LifecycleContext) -> None:
        """Capture infrastructure dependencies at pipeline start."""
        super().on_start(ctx)
        if ctx.landscape is None:
            raise FrameworkBugError("WebScrapeTransform requires landscape — orchestrator must inject it before on_start().")
        if ctx.rate_limit_registry is None:
            raise FrameworkBugError("WebScrapeTransform requires rate_limit_registry — orchestrator must inject it before on_start().")
        if ctx.payload_store is None:
            raise FrameworkBugError("WebScrapeTransform requires payload_store — orchestrator must configure it before on_start().")
        self._recorder = ctx.landscape
        self._payload_store = ctx.payload_store
        self._limiter = ctx.rate_limit_registry
        self._telemetry_emit = ctx.telemetry_emit

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Fetch URL and enrich row with content and fingerprint.

        Args:
            row: Input row (PipelineRow guaranteed by engine)
            ctx: Transform context with token, state_id, run_id

        Returns:
            TransformResult.success() with enriched row, or
            TransformResult.error() for non-retryable failures

        Raises:
            WebScrapeError: For retryable failures (5xx, 429, network)
                Engine RetryManager handles these with exponential backoff
        """
        url = row[self._url_field]

        # Validate URL and pin resolved IP (SSRF prevention with DNS rebinding defense)
        try:
            safe_request = validate_url_for_ssrf(url, allowed_ranges=self._allowed_ranges)
        except (SSRFBlockedError, SSRFNetworkError, TypeError) as e:
            # Security violations, DNS failures, and invalid url types (e.g. None)
            # are non-retryable
            return TransformResult.error(
                {
                    "reason": "validation_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

        # Fetch URL using pinned IP (prevents DNS rebinding between validation and fetch)
        try:
            response, final_hostname_url, call = self._fetch_url(safe_request, ctx)
            final_resolved_ip = _final_response_ip(response)
        except WebScrapeError as e:
            if e.retryable:
                # Re-raise retryable errors for engine RetryManager
                raise
            # Non-retryable errors return error result
            return TransformResult.error(
                {
                    "reason": "api_error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

        # Extract content — response.text is Tier 3 (external data), validate at boundary
        try:
            content = extract_content(
                response.text,
                format=self._format,
                strip_elements=self._strip_elements,
                text_separator=self._text_separator,
            )
        except (ValueError, UnicodeDecodeError, UnicodeEncodeError, RuntimeError) as e:
            return TransformResult.error(
                {
                    "reason": "content_extraction_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "url": safe_request.original_url,
                }
            )

        # Compute fingerprint
        fingerprint = compute_fingerprint(content, mode=self._fingerprint_mode)

        # Field collision check already done before fetch — no need to re-check here.

        # Hashes from audit trail — request and response blobs are already stored
        # by AuditedHTTPClient via recorder.record_call()
        if call.request_ref is None or call.response_ref is None:
            raise FrameworkBugError(
                "AuditedHTTPClient returned a Call with no request_ref/response_ref — "
                "PayloadStore must be configured for "
                "hash-based audit provenance in WebScrapeTransform."
            )
        request_hash = call.request_ref
        response_raw_hash = call.response_ref

        # Store processed content via PayloadStore (transform-produced artifact).
        # payload_store is guaranteed non-None by on_start() validation.
        response_processed_hash = self._payload_store.store(content.encode())

        # Enrich row with scraped data — operational fields only
        # Use explicit to_dict() conversion (PipelineRow guaranteed by engine)
        output = row.to_dict()
        output[self._content_field] = content
        output[self._fingerprint_field] = fingerprint
        output["fetch_status"] = response.status_code
        output["fetch_url_final"] = final_hostname_url
        output["fetch_url_final_ip"] = final_resolved_ip

        # Propagate contract so FIXED schemas can access fields added during enrichment
        output_contract = narrow_contract_to_output(
            input_contract=row.contract,
            output_row=output,
        )
        output_contract = self._apply_declared_output_field_contracts(output_contract)
        output_contract = self._align_output_contract(output_contract)

        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "enriched",
                "fields_added": [self._content_field, self._fingerprint_field],
                "metadata": {
                    "fetch_request_hash": request_hash,
                    "fetch_response_raw_hash": response_raw_hash,
                    "fetch_response_processed_hash": response_processed_hash,
                },
            },
        )

    def _fetch_url(self, safe_request: SSRFSafeRequest, ctx: TransformContext) -> tuple[httpx.Response, str, Call]:
        """Fetch URL using SSRF-safe IP pinning with audit recording.

        Args:
            safe_request: Pre-validated SSRFSafeRequest with pinned IP
            ctx: Plugin context

        Returns:
            Tuple of (httpx.Response, final hostname URL as string, Call).
            The hostname URL is the logical URL after redirects — distinct
            from response.url which is IP-based due to SSRF pinning.
            The Call contains request_ref and response_ref blob hashes.

        Raises:
            WebScrapeError: For retryable or non-retryable failures
        """
        # Infrastructure captured in on_start()
        if ctx.state_id is None:
            raise FrameworkBugError("ctx.state_id not set by executor — executor must set state_id before calling process().")
        limiter = self._limiter.get_limiter("web_scrape")

        # Create audited client (records to Landscape)
        client = AuditedHTTPClient(
            execution=self._recorder,
            state_id=ctx.state_id,
            run_id=ctx.run_id,
            telemetry_emit=self._telemetry_emit,
            timeout=self._timeout,
            limiter=limiter,
            token_id=ctx.token.token_id if ctx.token is not None else None,
        )

        # Add responsible scraping headers
        headers = {
            "X-Abuse-Contact": self._abuse_contact,
            "X-Scraping-Reason": self._scraping_reason,
        }

        try:
            response, final_hostname_url, call = client.get_ssrf_safe(
                safe_request,
                headers=headers,
                follow_redirects=True,
                allowed_ranges=self._allowed_ranges,
            )

            # Check status code and raise appropriate errors
            url = safe_request.original_url
            if response.status_code == 404:
                raise NotFoundError(f"HTTP 404: {url}")
            elif response.status_code == 403:
                raise ForbiddenError(f"HTTP 403: {url}")
            elif response.status_code == 401:
                raise UnauthorizedError(f"HTTP 401: {url}")
            elif response.status_code == 429:
                raise RateLimitError(f"HTTP 429: {url}")
            elif 500 <= response.status_code < 600:
                raise ServerError(f"HTTP {response.status_code}: {url}")
            elif 300 <= response.status_code < 400:
                # Unresolved redirect (e.g. 3xx without Location header) — treat as error
                raise InvalidURLError(f"Unresolved redirect HTTP {response.status_code}: {url} (missing or empty Location header)")

            return response, final_hostname_url, call

        except httpx.TimeoutException as e:
            raise NetworkError(f"Timeout fetching {safe_request.original_url}: {e}") from e
        except httpx.ConnectError as e:
            raise NetworkError(f"Connection error fetching {safe_request.original_url}: {e}") from e
        except SSRFBlockedError as e:
            # Redirect hop resolved to a blocked IP — non-retryable security violation
            from elspeth.plugins.transforms.web_scrape_errors import SSRFBlockedError as WSSRFBlockedError

            raise WSSRFBlockedError(f"SSRF blocked during redirect: {safe_request.original_url}: {e}") from e
        except SSRFNetworkError as e:
            # DNS resolution failed during redirect hop
            raise NetworkError(f"DNS resolution failed during redirect: {safe_request.original_url}: {e}") from e
        except httpx.TooManyRedirects as e:
            raise InvalidURLError(f"Too many redirects: {safe_request.original_url}: {e}") from e
        except httpx.RequestError as e:
            raise NetworkError(f"HTTP request error fetching {safe_request.original_url}: {e}") from e
        finally:
            client.close()

    def close(self) -> None:
        """Release resources."""
        pass
