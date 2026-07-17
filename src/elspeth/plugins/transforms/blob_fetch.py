"""Fetch an HTTP(S) document into the run payload store as a blob reference."""

from __future__ import annotations

import ipaddress
from ipaddress import IPv4Network, IPv6Network
from typing import Annotated, Any, Literal, cast

import httpx
from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.audit import Call
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.contract_propagation import narrow_contract_to_output
from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.wire_visible_identity import is_wire_visible_placeholder
from elspeth.core.security.web import NetworkError as SSRFNetworkError
from elspeth.core.security.web import SSRFBlockedError, SSRFSafeRequest, validate_url_for_ssrf
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.clients.fingerprinting import fingerprint_url
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient, HTTPResponseBodyTooLargeError
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.transforms.web_scrape_errors import (
    BodyTooLargeError,
    ClientError,
    ForbiddenError,
    InvalidURLError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ServerError,
    UnauthorizedError,
    WebScrapeError,
)

DEFAULT_ALLOWED_CONTENT_TYPES: tuple[str, ...] = (
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/json",
    "application/jsonl",
    "application/x-jsonlines",
    "application/x-ndjson",
    "text/jsonl",
    "application/xml",
    "text/xml",
)


def _validate_cidr_entry(entry: str) -> str:
    try:
        ipaddress.ip_network(entry, strict=False)
    except ValueError as exc:
        raise ValueError(f"Invalid CIDR in allowed_hosts: {entry!r}: {exc}") from exc
    return entry


CidrStr = Annotated[str, AfterValidator(_validate_cidr_entry)]


class BlobFetchHTTPConfig(BaseModel):
    """HTTP policy for fetching remote files into payload-store blobs."""

    model_config = {"extra": "forbid"}

    abuse_contact: str = Field(description="Email for abuse reports, sent as an HTTP header.")
    fetch_reason: str = Field(description="Why this pipeline is fetching remote files; recorded in audit and sent as an HTTP header.")
    timeout: int = Field(default=30, gt=0, description="Request timeout in seconds.")
    max_body_bytes: int = Field(default=10 * 1024 * 1024, gt=0, description="Maximum response body size in bytes.")
    allowed_hosts: Literal["public_only", "allow_private"] | Annotated[list[CidrStr], Field(min_length=1)] = Field(
        default="public_only",
        description="SSRF allowlist: 'public_only', 'allow_private', or explicit CIDR ranges.",
    )

    @field_validator("abuse_contact", "fetch_reason")
    @classmethod
    def _validate_wire_visible_header(cls, value: str, info: Any) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        if is_wire_visible_placeholder(value):
            raise ValueError(
                f"{info.field_name} must be supplied by the operator or deployment identity; "
                "placeholder values are not valid for wire-visible HTTP headers"
            )
        if not value.isascii():
            bad_index = next(i for i, char in enumerate(value) if not char.isascii())
            bad_char = value[bad_index]
            raise ValueError(
                f"{info.field_name} contains a non-ASCII character {bad_char!r} "
                f"(U+{ord(bad_char):04X}) at position {bad_index}; this value is sent "
                "verbatim as an HTTP request header and must be ASCII-encodable."
            )
        return value


class BlobFetchConfig(TransformDataConfig):
    """Configuration for blob_fetch."""

    url_field: str = Field(description="Input row field containing the absolute HTTP(S) URL to fetch.")
    blob_ref_field: str = Field(default="blob_ref", description="Output field receiving the payload-store content hash.")
    content_type_field: str = Field(default="blob_content_type", description="Output field receiving the normalized response Content-Type.")
    size_bytes_field: str = Field(default="blob_size_bytes", description="Output field receiving the response body size.")
    sha256_field: str = Field(default="blob_sha256", description="Output field receiving the response body SHA-256 hex digest.")
    fetch_status_field: str = Field(default="fetch_status", description="Output field receiving the HTTP status code.")
    fetch_url_final_field: str = Field(
        default="fetch_url_final", description="Output field receiving the final logical URL after redirects."
    )
    fetch_url_final_ip_field: str = Field(default="fetch_url_final_ip", description="Output field receiving the final resolved IP.")
    allowed_content_types: tuple[str, ...] = Field(
        default=DEFAULT_ALLOWED_CONTENT_TYPES,
        min_length=1,
        description="Exact normalized response Content-Type values accepted for storage.",
    )
    http: BlobFetchHTTPConfig = Field(description="HTTP fetching policy, timeout, contact, and host allowlist settings.")

    @field_validator("url_field")
    @classmethod
    def _reject_empty_input_field(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("url_field must not be empty")
        return value.strip()

    @field_validator(
        "blob_ref_field",
        "content_type_field",
        "size_bytes_field",
        "sha256_field",
        "fetch_status_field",
        "fetch_url_final_field",
        "fetch_url_final_ip_field",
    )
    @classmethod
    def _validate_output_field(cls, value: str, info: Any) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must not be empty")
        if not stripped.isidentifier():
            raise ValueError(f"{info.field_name} must be a valid Python identifier, got {value!r}")
        return stripped

    @field_validator("allowed_content_types")
    @classmethod
    def _normalize_allowed_content_types(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.split(";", 1)[0].strip().lower() for value in values)
        if any(not value for value in normalized):
            raise ValueError("allowed_content_types entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed_content_types entries must be unique after normalization")
        return normalized

    @property
    def declared_input_fields(self) -> frozenset[str]:
        return super().declared_input_fields | frozenset({self.url_field})

    @model_validator(mode="after")
    def _reject_output_collisions(self) -> BlobFetchConfig:
        output_fields = (
            self.blob_ref_field,
            self.content_type_field,
            self.size_bytes_field,
            self.sha256_field,
            self.fetch_status_field,
            self.fetch_url_final_field,
            self.fetch_url_final_ip_field,
        )
        duplicates = sorted({field for field in output_fields if output_fields.count(field) > 1})
        if duplicates:
            raise ValueError(f"Output fields must be unique; duplicates: {duplicates!r}")
        if self.url_field in output_fields:
            raise ValueError(f"url_field {self.url_field!r} collides with a blob_fetch output field")
        return self


def _parse_allowed_ranges(entries: list[str]) -> tuple[IPv4Network | IPv6Network, ...]:
    return tuple(ipaddress.ip_network(entry, strict=False) for entry in entries)


def _final_response_ip(response: httpx.Response) -> str:
    try:
        final_host = response.request.url.host
    except RuntimeError as exc:
        raise FrameworkBugError("SSRF-safe HTTP response has no request; cannot record final resolved IP.") from exc

    if final_host is None:
        raise FrameworkBugError("SSRF-safe HTTP response request URL has no host; cannot record final resolved IP.")
    try:
        ipaddress.ip_address(final_host)
    except ValueError as exc:
        raise FrameworkBugError(
            f"SSRF-safe HTTP response request host {final_host!r} is not an IP address; "
            "AuditedHTTPClient must return the IP-pinned final request."
        ) from exc
    return final_host


def _normalized_content_type(response: httpx.Response) -> str:
    return cast(str, response.headers.get("content-type", "")).split(";", 1)[0].strip().lower()


def _blob_fetch_added_output_fields(cfg: BlobFetchConfig) -> tuple[FieldDefinition, ...]:
    return (
        FieldDefinition(name=cfg.blob_ref_field, field_type="str", required=True),
        FieldDefinition(name=cfg.content_type_field, field_type="str", required=True),
        FieldDefinition(name=cfg.size_bytes_field, field_type="int", required=True),
        FieldDefinition(name=cfg.sha256_field, field_type="str", required=True),
        FieldDefinition(name=cfg.fetch_status_field, field_type="int", required=True),
        FieldDefinition(name=cfg.fetch_url_final_field, field_type="str", required=True),
        FieldDefinition(name=cfg.fetch_url_final_ip_field, field_type="str", required=True),
    )


def _build_blob_fetch_output_schema_config(schema_config: SchemaConfig, cfg: BlobFetchConfig) -> SchemaConfig:
    field_by_name: dict[str, FieldDefinition] = {}
    if schema_config.fields is not None:
        field_by_name.update((field.name, field) for field in schema_config.fields)

    added_fields = _blob_fetch_added_output_fields(cfg)
    field_by_name.update((field.name, field) for field in added_fields)

    base_guaranteed = set(schema_config.guaranteed_fields or ())
    output_guaranteed = base_guaranteed | {field.name for field in added_fields}

    return SchemaConfig(
        mode=schema_config.mode if schema_config.fields is not None else "flexible",
        fields=tuple(field_by_name.values()),
        guaranteed_fields=tuple(sorted(output_guaranteed)),
        audit_fields=schema_config.audit_fields,
        required_fields=schema_config.required_fields,
    )


class BlobFetch(BaseTransform):
    """Fetch an HTTP(S) URL into the run payload store and emit a blob reference."""

    name = "blob_fetch"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:2c5a53cc0e54a008"
    config_model = BlobFetchConfig
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "schema": {"mode": "observed"},
            "url_field": "blob_fetch_probe_url",
            "http": {
                "abuse_contact": "invariants@example.com",
                "fetch_reason": "ADR-009 invariant probe",
                "allowed_hosts": ["93.184.216.34/32"],
            },
        }

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        cfg = BlobFetchConfig.from_dict(options, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)

        self._url_field = cfg.url_field
        self._blob_ref_field = cfg.blob_ref_field
        self._content_type_field = cfg.content_type_field
        self._size_bytes_field = cfg.size_bytes_field
        self._sha256_field = cfg.sha256_field
        self._fetch_status_field = cfg.fetch_status_field
        self._fetch_url_final_field = cfg.fetch_url_final_field
        self._fetch_url_final_ip_field = cfg.fetch_url_final_ip_field
        self._allowed_content_types = frozenset(cfg.allowed_content_types)

        self._abuse_contact = cfg.http.abuse_contact
        self._fetch_reason = cfg.http.fetch_reason
        self._timeout = cfg.http.timeout
        self._max_body_bytes = cfg.http.max_body_bytes

        allowed_hosts = cfg.http.allowed_hosts
        if allowed_hosts == "public_only":
            self._allowed_ranges: tuple[IPv4Network | IPv6Network, ...] = ()
        elif allowed_hosts == "allow_private":
            self._allowed_ranges = (
                ipaddress.ip_network("0.0.0.0/0"),
                ipaddress.ip_network("::/0"),
            )
        else:
            self._allowed_ranges = _parse_allowed_ranges(allowed_hosts)

        output_fields = [field.name for field in _blob_fetch_added_output_fields(cfg)]
        self.declared_output_fields = frozenset(output_fields)

        self.input_schema = create_schema_from_config(cfg.schema_config, "BlobFetchInput", allow_coercion=False)
        self._output_schema_config = _build_blob_fetch_output_schema_config(cfg.schema_config, cfg)
        self.output_schema = create_schema_from_config(self._output_schema_config, "BlobFetchOutput", allow_coercion=False)

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Fetch an HTTP(S) URL into the run payload store and emit a blob reference plus fetch metadata.",
                composer_hints=(
                    "Use blob_fetch when rows contain document URLs and downstream parser transforms should consume blob_ref, not raw bytes.",
                    "blob_fetch does not parse content; chain blob_csv_expand, future blob_json_expand, or another blob parser after it.",
                    "allowed_content_types is a strict exact MIME allowlist; add operator-approved types explicitly.",
                    "http.abuse_contact and http.fetch_reason are mandatory and sent as wire-visible headers.",
                ),
            )
        return None

    def on_start(self, ctx: LifecycleContext) -> None:
        super().on_start(ctx)
        if ctx.landscape is None:
            raise FrameworkBugError("BlobFetch requires landscape — orchestrator must inject it before on_start().")
        if ctx.rate_limit_registry is None:
            raise FrameworkBugError("BlobFetch requires rate_limit_registry — orchestrator must inject it before on_start().")
        if ctx.payload_store is None:
            raise FrameworkBugError("BlobFetch requires payload_store — orchestrator must configure it before on_start().")
        self._recorder = ctx.landscape
        self._payload_store = ctx.payload_store
        self._limiter = ctx.rate_limit_registry
        self._telemetry_emit = ctx.telemetry_emit

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        try:
            url = row[self._url_field]
            safe_request = validate_url_for_ssrf(url, allowed_ranges=self._allowed_ranges)
        except (KeyError, SSRFBlockedError, SSRFNetworkError, TypeError) as exc:
            return TransformResult.error(
                {
                    "reason": "validation_failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

        try:
            response, final_hostname_url, call = self._fetch_url(safe_request, ctx)
            final_resolved_ip = _final_response_ip(response)
        except BodyTooLargeError as exc:
            safe_url = fingerprint_url(safe_request.original_url)
            return TransformResult.error(
                {
                    "reason": "body_too_large",
                    "error": f"response body {exc.body_size} bytes exceeds max_body_bytes {exc.max_body_bytes} for {safe_url}",
                    "url": safe_url,
                    "body_size": exc.body_size,
                    "max_body_bytes": exc.max_body_bytes,
                }
            )
        except WebScrapeError as exc:
            if exc.retryable:
                raise
            return TransformResult.error(
                {
                    "reason": "api_error",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

        content_type = _normalized_content_type(response)
        safe_url = fingerprint_url(safe_request.original_url)
        if content_type not in self._allowed_content_types:
            return TransformResult.error(
                {
                    "reason": "unsupported_content_type",
                    "error": (
                        f"content-type {response.headers.get('content-type', '')!r} returned by {safe_url}; "
                        f"allowed values are {sorted(self._allowed_content_types)!r}"
                    ),
                    "content_type": response.headers.get("content-type", ""),
                    "url": safe_url,
                }
            )

        body = response.content
        body_size = len(body)
        if body_size > self._max_body_bytes:
            return TransformResult.error(
                {
                    "reason": "body_too_large",
                    "error": f"response body {body_size} bytes exceeds max_body_bytes {self._max_body_bytes} for {safe_url}",
                    "body_size": body_size,
                    "max_body_bytes": self._max_body_bytes,
                    "url": safe_url,
                }
            )

        blob_ref = self._payload_store.store(body)
        if call.request_ref is None or call.response_ref is None:
            raise FrameworkBugError(
                "AuditedHTTPClient returned a Call with no request_ref/response_ref — "
                "PayloadStore must be configured for hash-based fetch provenance in BlobFetch."
            )

        output = row.to_dict()
        output[self._blob_ref_field] = blob_ref
        output[self._content_type_field] = content_type
        output[self._size_bytes_field] = body_size
        output[self._sha256_field] = blob_ref
        output[self._fetch_status_field] = response.status_code
        output[self._fetch_url_final_field] = fingerprint_url(final_hostname_url)
        output[self._fetch_url_final_ip_field] = final_resolved_ip

        output_contract = narrow_contract_to_output(input_contract=row.contract, output_row=output)
        output_contract = self._apply_declared_output_field_contracts(output_contract)
        output_contract = self._align_output_contract(output_contract)

        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "enriched",
                "fields_added": sorted(self.declared_output_fields),
                "metadata": {
                    "fetch_request_hash": call.request_ref,
                    "fetch_response_raw_hash": call.response_ref,
                    "fetch_payload_hash": blob_ref,
                },
            },
        )

    def _fetch_url(self, safe_request: SSRFSafeRequest, ctx: TransformContext) -> tuple[httpx.Response, str, Call]:
        if ctx.state_id is None:
            raise FrameworkBugError("ctx.state_id not set by executor — executor must set state_id before calling process().")
        safe_url = fingerprint_url(safe_request.original_url)
        limiter = self._limiter.get_limiter("blob_fetch")
        client = AuditedHTTPClient(
            execution=self._recorder,
            state_id=ctx.state_id,
            run_id=ctx.run_id,
            telemetry_emit=self._telemetry_emit,
            timeout=self._timeout,
            limiter=limiter,
            token_id=ctx.token.token_id if ctx.token is not None else None,
            max_response_body_bytes=self._max_body_bytes,
        )
        headers = {
            "X-Abuse-Contact": self._abuse_contact,
            "X-Fetch-Reason": self._fetch_reason,
        }

        try:
            response, final_hostname_url, call = client.get_ssrf_safe(
                safe_request,
                headers=headers,
                follow_redirects=True,
                allowed_ranges=self._allowed_ranges,
            )
            if response.status_code == 404:
                raise NotFoundError(f"HTTP 404: {safe_url}")
            if response.status_code == 403:
                raise ForbiddenError(f"HTTP 403: {safe_url}")
            if response.status_code == 401:
                raise UnauthorizedError(f"HTTP 401: {safe_url}")
            if response.status_code == 429:
                raise RateLimitError(f"HTTP 429: {safe_url}")
            if 500 <= response.status_code < 600:
                raise ServerError(f"HTTP {response.status_code}: {safe_url}")
            if 300 <= response.status_code < 400:
                raise InvalidURLError(f"Unresolved redirect HTTP {response.status_code}: {safe_url} (missing or empty Location header)")
            if 400 <= response.status_code < 500:
                raise ClientError(f"HTTP {response.status_code}: {safe_url}", retryable=response.status_code == 408)
            return response, final_hostname_url, call
        except httpx.TimeoutException as exc:
            raise NetworkError(f"Timeout fetching {safe_url}") from exc
        except httpx.ConnectError as exc:
            raise NetworkError(f"Connection error fetching {safe_url}") from exc
        except HTTPResponseBodyTooLargeError as exc:
            raise BodyTooLargeError(
                f"response body {exc.body_size} bytes exceeds max_body_bytes {exc.max_body_bytes} for {safe_url}",
                body_size=exc.body_size,
                max_body_bytes=exc.max_body_bytes,
            ) from exc
        except SSRFBlockedError as exc:
            from elspeth.plugins.transforms.web_scrape_errors import SSRFBlockedError as WSSRFBlockedError

            raise WSSRFBlockedError(f"SSRF blocked during redirect while fetching {safe_url}") from exc
        except SSRFNetworkError as exc:
            raise NetworkError(f"DNS resolution failed during redirect while fetching {safe_url}") from exc
        except httpx.TooManyRedirects as exc:
            raise InvalidURLError(f"Too many redirects while fetching {safe_url}") from exc
        except httpx.RequestError as exc:
            raise NetworkError(f"HTTP request error fetching {safe_url} ({type(exc).__name__})") from exc
        finally:
            client.close()

    def close(self) -> None:
        pass
