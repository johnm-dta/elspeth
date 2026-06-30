"""Azure AI Document Intelligence enrichment transform.

Submits a document reference (URL or base64) per row to Azure Document
Intelligence, waits for the asynchronous analysis (POST 202 + Operation-Location
poll), and enriches the row with extracted content and structured facets.

All HTTP flows through AuditedHTTPClient (full request/response audit, header
fingerprinting so the api-key is never stored raw, telemetry, rate limiting).
GA api-version 2024-11-30. See
docs/superpowers/specs/2026-06-30-azure-document-intelligence-transform-design.md.

SECURITY:
- A SAS token embedded in a ``urlSource`` value is forwarded to Azure and
  recorded in the audited request blob (operator data).
- ``base64Source`` records the entire document verbatim in the audited request
  blob (~1.33x size). Operators with PII-sensitive documents should weigh their
  audit-retention policy.
- The polled ``Operation-Location`` URL carries our api-key header; we refuse to
  follow it unless its host matches the configured ``endpoint`` host.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Self
from urllib.parse import urlencode, urlparse

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.contract_propagation import narrow_contract_to_output
from elspeth.contracts.enums import AuditCharacteristic
from elspeth.contracts.errors import FrameworkBugError, PluginRetryableError, is_capacity_error
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient, HTTPResponseBodyTooLargeError
from elspeth.plugins.infrastructure.clients.json_utils import parse_json_strict
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.infrastructure.telemetry import make_warn_telemetry_before_start
from elspeth.plugins.transforms.azure.document_intelligence_result import (
    count_pages,
    extract_content,
    extract_facet_list,
    operation_id_from_url,
    operation_location_host_matches,
)
from elspeth.plugins.transforms.azure.errors import MalformedResponseError

logger = structlog.get_logger(__name__)
_warn_telemetry_before_start = make_warn_telemetry_before_start(logger)

_HTTP_TIMEOUT_HEADROOM_SECONDS = 30.0

_SUPPORTED_API_VERSIONS: frozenset[str] = frozenset({"2024-11-30"})

# config attr on ExtractFields -> Azure analyzeResult key
_FACET_AZURE_KEYS: dict[str, str] = {
    "pages": "pages",
    "tables": "tables",
    "key_value_pairs": "keyValuePairs",
    "paragraphs": "paragraphs",
    "documents": "documents",
    "languages": "languages",
    "styles": "styles",
    "figures": "figures",
    "sections": "sections",
}

_KNOWN_FEATURES: frozenset[str] = frozenset(
    {"ocrHighResolution", "languages", "barcodes", "formulas", "keyValuePairs", "styleFont", "queryFields"}
)

_MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._~-]{1,63}$")
_PAGES_PATTERN = re.compile(r"^(\d+(-\d+)?)(,\s*(\d+(-\d+)?))*$")


class ExtractFields(BaseModel):
    """Facet -> output-field-name map. Only set facets are emitted as row fields.

    ``figures`` reads ``analyzeResult.figures`` metadata (regions/spans), which is
    distinct from the ``output=figures`` artifact-generation option (out of scope).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pages: str | None = None
    tables: str | None = None
    key_value_pairs: str | None = None
    paragraphs: str | None = None
    documents: str | None = None
    languages: str | None = None
    styles: str | None = None
    figures: str | None = None
    sections: str | None = None


class AzureDocumentIntelligenceConfig(TransformDataConfig):
    """Configuration for the azure_document_intelligence transform."""

    endpoint: str = Field(..., description="Azure Document Intelligence endpoint URL (HTTPS).")
    api_key: str = Field(..., repr=False, description="Document Intelligence API key (Ocp-Apim-Subscription-Key).")
    api_version: str = Field("2024-11-30", description="REST api-version (GA 2024-11-30 only in v1).")
    model_id: str = Field(..., description="Prebuilt or custom model id, e.g. prebuilt-layout, prebuilt-invoice.")

    source_mode: str = Field(..., pattern=r"^(url|base64)$", description="How the document reference is read: 'url' or 'base64'.")
    source_field: str = Field(..., description="Row field holding the document URL (url mode) or base64 string (base64 mode).")
    max_base64_chars: int = Field(8_000_000, gt=0, description="Max length of a base64 source string (bounds the audited request blob).")

    content_field: str | None = Field(None, description="Row field for the full text/markdown content.")
    output_content_format: str = Field("text", pattern=r"^(text|markdown)$", description="outputContentFormat: text or markdown.")
    extract: ExtractFields = Field(default_factory=ExtractFields, description="Facet -> output-field-name map.")
    page_count_field: str | None = Field(None, description="Row field for the analyzed page count (int).")
    result_field: str | None = Field(
        None,
        description=(
            "Row field for the entire analyzeResult object (max fidelity). Stores the full "
            "result (~5-50 MB for large documents) as a row value; prefer content_field or "
            "specific extract facets for normal use."
        ),
    )

    pages: str | None = Field(None, description="1-based page range, e.g. '1-3,5,7-9'.")
    locale: str | None = Field(None, description="Locale hint, e.g. 'en-US'.")
    string_index_type: str = Field(
        "textElements",
        pattern=r"^(textElements|unicodeCodePoint|utf16CodeUnit)$",
        description="stringIndexType: textElements, unicodeCodePoint, or utf16CodeUnit.",
    )
    features: list[str] = Field(default_factory=list, description="Add-on analysis features.")
    query_fields: list[str] = Field(default_factory=list, description="Additional fields to extract (requires queryFields feature).")

    poll_interval_seconds: float = Field(1.0, gt=0, description="Initial poll delay.")
    poll_backoff_multiplier: float = Field(1.5, ge=1, description="Exponential poll backoff factor.")
    poll_max_interval_seconds: float = Field(10.0, gt=0, description="Poll backoff cap.")
    poll_timeout_seconds: float = Field(300.0, gt=0, description="Outer bound on the poll sequence.")
    request_timeout_seconds: float = Field(60.0, gt=0, description="Per HTTP call timeout.")
    max_response_body_bytes: int = Field(50_000_000, gt=0, description="Cap on the analyzeResult response body.")
    max_capacity_retry_seconds: int = Field(
        300, gt=0, description="Per-call 429/503 retry budget (default 300 bounds web-worker hold time)."
    )
    batch_wait_timeout_seconds: int = Field(3600, gt=0, description="Batch-mixin per-row wait timeout.")

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint(cls, v: str) -> str:
        from elspeth.plugins.infrastructure.url_validation import validate_credential_safe_https_url

        return validate_credential_safe_https_url(v, field_name="endpoint")

    @field_validator("api_key")
    @classmethod
    def _reject_empty_api_key(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("api_key must not be empty")
        return v

    @field_validator("api_version")
    @classmethod
    def _validate_api_version(cls, v: str) -> str:
        if v not in _SUPPORTED_API_VERSIONS:
            raise ValueError(f"api_version {v!r} is not supported. Supported: {sorted(_SUPPORTED_API_VERSIONS)}.")
        return v

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, v: str) -> str:
        if not _MODEL_ID_PATTERN.match(v):
            raise ValueError(f"model_id {v!r} must match {_MODEL_ID_PATTERN.pattern} (Azure model id rules).")
        return v

    @field_validator("source_field")
    @classmethod
    def _reject_empty_source_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_field must not be empty")
        return v

    @field_validator("pages")
    @classmethod
    def _validate_pages(cls, v: str | None) -> str | None:
        if v is not None and not _PAGES_PATTERN.match(v):
            raise ValueError(f"pages {v!r} must match {_PAGES_PATTERN.pattern} (e.g. '1-3,5').")
        return v

    @field_validator("features")
    @classmethod
    def _validate_features(cls, v: list[str]) -> list[str]:
        unknown = [feature for feature in v if feature not in _KNOWN_FEATURES]
        if unknown:
            raise ValueError(f"Unknown features {unknown}. Known: {sorted(_KNOWN_FEATURES)}.")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        names = self.all_output_field_names()
        if not names:
            raise ValueError(
                "At least one output target must be configured: content_field, page_count_field, result_field, or one entry in extract."
            )
        dupes = sorted({name for name in names if names.count(name) > 1})
        if dupes:
            raise ValueError(f"Duplicate output field names: {dupes}. Each output field must be unique.")
        has_qf_feature = "queryFields" in self.features
        if bool(self.query_fields) != has_qf_feature:
            raise ValueError("query_fields and the 'queryFields' feature must be set together (both or neither).")
        if self.poll_max_interval_seconds < self.poll_interval_seconds:
            raise ValueError("poll_max_interval_seconds must be >= poll_interval_seconds.")
        return self

    def configured_output_fields(self) -> dict[str, str]:
        """Map Azure analyzeResult facet key -> output row field, for set extract entries only."""
        result: dict[str, str] = {}
        for attr, azure_key in _FACET_AZURE_KEYS.items():
            field_name = getattr(self.extract, attr)
            if field_name is not None:
                result[azure_key] = field_name
        return result

    def all_output_field_names(self) -> list[str]:
        """Every output row field this transform may add (order: content, page_count, result, facets)."""
        names: list[str] = []
        if self.content_field is not None:
            names.append(self.content_field)
        if self.page_count_field is not None:
            names.append(self.page_count_field)
        if self.result_field is not None:
            names.append(self.result_field)
        names.extend(self.configured_output_fields().values())
        return names


class AzureDocumentIntelligence(BaseTransform, BatchTransformMixin):
    """Enrich rows with Azure Document Intelligence extraction (async analyze LRO).

    Each row supplies one document reference (URL or base64); the transform submits
    it, polls the long-running operation to completion, and adds the configured
    extraction facets to the row. Uses row-level pipelining (BatchTransformMixin):
    multiple rows are in flight concurrently with FIFO output ordering.
    """

    name = "azure_document_intelligence"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    # Placeholder must be a sha256: literal so the hash normalizer matches it; recomputed by scripts/cicd/plugin_hash.
    source_file_hash: str | None = "sha256:4799f6bfb0d43e52"
    config_model = AzureDocumentIntelligenceConfig
    passes_through_input = True
    creates_tokens = False
    discovery_secret_requirements: Mapping[str, tuple[str, ...]] = {
        "api_key": ("AZURE_DOCUMENT_INTELLIGENCE_KEY",),
    }
    audit_characteristics = frozenset({AuditCharacteristic.CREDENTIALS})
    capability_tags = ("azure", "document", "ocr", "enrichment", "http")

    usage_when_to_use = (
        "Use to turn documents (PDFs, images, office files) referenced per row "
        "into structured data — text/markdown content, tables, key-value pairs, "
        "and the typed fields of prebuilt or custom Document Intelligence models."
    )
    usage_when_not_to_use = (
        "Not for moderation or injection screening (use azure_content_safety / "
        "azure_prompt_shield), and not for plain web pages (use web_scrape). "
        "Documents must be reachable by a URL or supplied as a base64 string."
    )
    example_use = (
        "transform:\n"
        "  plugin: azure_document_intelligence\n"
        "  options:\n"
        "    endpoint: https://my-di.cognitiveservices.azure.com\n"
        "    api_key: ${AZURE_DOCUMENT_INTELLIGENCE_KEY}\n"
        "    model_id: prebuilt-layout\n"
        "    source_mode: url\n"
        "    source_field: document_url\n"
        "    content_field: di_content\n"
        "    output_content_format: markdown\n"
        "    extract: {tables: di_tables, key_value_pairs: di_kv}\n"
        "    schema: {mode: observed}"
    )

    _CAPACITY_RETRY_INITIAL_DELAY_SECONDS = 0.05
    _CAPACITY_RETRY_MAX_DELAY_SECONDS = 1.0

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 forward invariant (no network/credentials)."""
        return {
            "endpoint": "https://test.cognitiveservices.azure.com",
            "api_key": "test-key",
            "model_id": "prebuilt-layout",
            "source_mode": "url",
            "source_field": "doc_intelligence_probe_url",
            "content_field": "di_content",
            "schema": {"mode": "observed"},
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = AzureDocumentIntelligenceConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._cfg = cfg

        self._endpoint = cfg.endpoint.rstrip("/")
        self._api_key = cfg.api_key
        self._api_version = cfg.api_version
        self._model_id = cfg.model_id
        self._source_mode = cfg.source_mode
        self._source_field = cfg.source_field
        self._max_base64_chars = cfg.max_base64_chars
        self._content_field = cfg.content_field
        self._output_content_format = cfg.output_content_format
        self._page_count_field = cfg.page_count_field
        self._result_field = cfg.result_field
        self._facet_fields = cfg.configured_output_fields()  # azure_key -> row field
        self._request_timeout_seconds = cfg.request_timeout_seconds
        self._max_response_body_bytes = cfg.max_response_body_bytes
        self._max_capacity_retry_seconds = cfg.max_capacity_retry_seconds
        self._poll_interval_seconds = cfg.poll_interval_seconds
        self._poll_backoff_multiplier = cfg.poll_backoff_multiplier
        self._poll_max_interval_seconds = cfg.poll_max_interval_seconds
        self._poll_timeout_seconds = cfg.poll_timeout_seconds
        self._batch_wait_timeout_seconds = cfg.batch_wait_timeout_seconds
        self._effective_batch_wait_timeout_seconds = max(
            float(self._batch_wait_timeout_seconds),
            float(self._poll_timeout_seconds) + float(self._max_capacity_retry_seconds) + _HTTP_TIMEOUT_HEADROOM_SECONDS,
        )

        self.declared_output_fields = frozenset(cfg.all_output_field_names())

        schema_config = cfg.schema_config
        self.input_schema = create_schema_from_config(schema_config, "AzureDocumentIntelligenceInput", allow_coercion=False)
        self._output_schema_config = self._build_output_schema_config(schema_config)
        self.output_schema = create_schema_from_config(self._output_schema_config, "AzureDocumentIntelligenceOutput", allow_coercion=False)

        self._recorder: PluginAuditWriter | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = _warn_telemetry_before_start
        self._limiter: Any = None
        self._http_clients: dict[str, Any] = {}
        self._http_clients_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._batch_initialized = False

    def _default_request_headers(self) -> dict[str, str]:
        """Auth headers for the audited client. Isolated so AAD bearer auth is a clean later add."""
        return {"Ocp-Apim-Subscription-Key": self._api_key}

    def on_start(self, ctx: LifecycleContext) -> None:
        super().on_start(ctx)
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        self._limiter = ctx.rate_limit_registry.get_limiter(self.name) if ctx.rate_limit_registry is not None else None

    def connect_output(self, output: OutputPort, max_pending: int = 30) -> None:
        if self._batch_initialized:
            raise RuntimeError("connect_output() already called")
        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,
            batch_wait_timeout=self._effective_batch_wait_timeout_seconds,
        )
        self._batch_initialized = True

    def accept(self, row: PipelineRow, ctx: TransformContext) -> None:
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept().")
        self.accept_row(row, ctx, self._process_row)

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        raise NotImplementedError(f"{self.__class__.__name__} uses row-level pipelining. Use accept() instead of process().")

    def _get_http_client(self, state_id: str, *, token_id: str | None = None) -> Any:
        """Get or create the audited HTTP client for a state_id (cached for call_index continuity)."""
        with self._http_clients_lock:
            if state_id not in self._http_clients:
                if self._recorder is None:
                    raise RuntimeError(f"{self.name}: recorder not initialized — call on_start() before processing")
                self._http_clients[state_id] = AuditedHTTPClient(
                    execution=self._recorder,
                    state_id=state_id,
                    run_id=self._run_id,
                    telemetry_emit=self._telemetry_emit,
                    timeout=self._request_timeout_seconds,
                    headers=self._default_request_headers(),
                    limiter=self._limiter,
                    token_id=token_id,
                    max_response_body_bytes=self._max_response_body_bytes,
                )
            return self._http_clients[state_id]

    def close(self) -> None:
        self._shutdown.set()
        if self._batch_initialized:
            self.shutdown_batch_processing()
        with self._http_clients_lock:
            for client in self._http_clients.values():
                client.close()
            self._http_clients.clear()
        self._recorder = None

    # ── Per-row processing ────────────────────────────────────────────────

    def _process_row(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        state_id = ctx.state_id
        if state_id is None:
            raise RuntimeError("state_id is required for batch processing.")
        token_id = ctx.token.token_id if ctx.token is not None else None
        try:
            return self._process_single_with_state(row, state_id, token_id=token_id)
        finally:
            with self._http_clients_lock:
                client = self._http_clients.pop(state_id, None)
            if client is not None:
                client.close()

    def _process_single_with_state(self, row: PipelineRow, state_id: str, *, token_id: str | None = None) -> TransformResult:
        """Submit the document, poll the LRO to completion, and enrich the row.

        ``capacity_deadline`` is a single per-row total-budget anchor shared by the
        submit POST and every poll GET (it is NOT a per-phase budget): the
        ``max_capacity_retry_seconds`` cap applies to total 429/503 wait across the
        whole row. Poll GETs further clamp it to ``min(capacity_deadline,
        poll_deadline)`` so capacity retries never outlive the outer poll budget.
        """
        if self._source_field not in row:
            return TransformResult.error({"reason": "missing_field", "field": self._source_field}, retryable=False)
        ref = row[self._source_field]
        if not isinstance(ref, str) or not ref.strip():
            return TransformResult.error(
                {"reason": "non_string_field", "field": self._source_field, "actual_type": type(ref).__name__},
                retryable=False,
            )

        if self._source_mode == "url":
            parsed = urlparse(ref)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                return TransformResult.error(
                    {"reason": "invalid_input", "field": self._source_field, "error_type": "invalid_document_url"},
                    retryable=False,
                )
            body: dict[str, str] = {"urlSource": ref}
        else:
            if len(ref) > self._max_base64_chars:
                return TransformResult.error(
                    {
                        "reason": "invalid_input",
                        "field": self._source_field,
                        "error_type": "base64_too_large",
                        "message": f"base64 source length {len(ref)} exceeds max {self._max_base64_chars}",
                    },
                    retryable=False,
                )
            body = {"base64Source": ref}

        started_at = time.monotonic()
        capacity_deadline = started_at + float(self._max_capacity_retry_seconds)

        submission = self._submit(body, state_id, token_id=token_id, capacity_deadline=capacity_deadline, started_at=started_at)
        if isinstance(submission, TransformResult):
            return submission
        operation_url, retry_after = submission

        analyze = self._poll(
            operation_url,
            state_id,
            token_id=token_id,
            retry_after=retry_after,
            capacity_deadline=capacity_deadline,
            started_at=started_at,
        )
        if isinstance(analyze, TransformResult):
            return analyze

        return self._build_enriched_result(row, analyze, operation_url=operation_url)

    def _analyze_url(self) -> str:
        params: list[tuple[str, str]] = [("_overload", "analyzeDocument"), ("api-version", self._api_version)]
        if self._output_content_format != "text":
            params.append(("outputContentFormat", self._output_content_format))
        if self._cfg.pages is not None:
            params.append(("pages", self._cfg.pages))
        if self._cfg.locale is not None:
            params.append(("locale", self._cfg.locale))
        params.append(("stringIndexType", self._cfg.string_index_type))
        if self._cfg.features:
            params.append(("features", ",".join(self._cfg.features)))
        if self._cfg.query_fields:
            params.append(("queryFields", ",".join(self._cfg.query_fields)))
        query = urlencode(params)
        return f"{self._endpoint}/documentintelligence/documentModels/{self._model_id}:analyze?{query}"

    def _http_call_with_capacity_retry(self, do_call: Callable[[], Any], *, deadline: float, started_at: float, error_type: str) -> Any:
        """Run one HTTP call with capacity (429/503) retry. Returns the response or a TransformResult.

        ``deadline`` is the total-budget anchor (``min(capacity_deadline, poll_deadline)``
        when called from ``_poll``). Non-capacity HTTP errors map to the ``api_error``
        category with ``error_type`` as the sub-label. Raises PluginRetryableError on
        network errors (engine retries the row).
        """
        delay = self._CAPACITY_RETRY_INITIAL_DELAY_SECONDS
        while True:
            if self._shutdown.is_set():
                return self._shutdown_result(started_at)
            try:
                response = do_call()
                response.raise_for_status()
                return response
            except HTTPResponseBodyTooLargeError as e:
                # Distinct exception class (subclass of httpx.HTTPError, NOT HTTPStatusError /
                # RequestError). AuditedHTTPClient raises it when a response body exceeds
                # max_response_body_bytes. If it escaped here it would propagate past the
                # executor and cancel the WHOLE batch; convert to a per-row error instead.
                return TransformResult.error(
                    {"reason": "body_too_large", "body_size": e.body_size, "max_body_bytes": e.max_body_bytes},
                    retryable=False,
                )
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if not is_capacity_error(status_code):
                    return TransformResult.error(
                        {"reason": "api_error", "error_type": error_type, "status_code": status_code}, retryable=False
                    )
                now = time.monotonic()
                if now >= deadline:
                    return self._retry_timeout_result(status_code, started_at)
                sleep_seconds = min(delay, deadline - now)
                if sleep_seconds > 0 and self._shutdown.wait(timeout=sleep_seconds):
                    return self._shutdown_result(started_at)
                if time.monotonic() >= deadline:
                    return self._retry_timeout_result(status_code, started_at)
                delay = min(delay * 2.0, self._CAPACITY_RETRY_MAX_DELAY_SECONDS)
            except httpx.RequestError as e:
                raise PluginRetryableError(f"Azure Document Intelligence network error: {e}", retryable=True) from e

    def _submit(
        self,
        body: dict[str, str],
        state_id: str,
        *,
        token_id: str | None,
        capacity_deadline: float,
        started_at: float,
    ) -> tuple[str, float | None] | TransformResult:
        client = self._get_http_client(state_id, token_id=token_id)
        url = self._analyze_url()

        def do_call() -> Any:
            return client.post(url, json=body, timeout=self._request_timeout_seconds)

        resp = self._http_call_with_capacity_retry(do_call, deadline=capacity_deadline, started_at=started_at, error_type="submit_rejected")
        if isinstance(resp, TransformResult):
            return resp
        if resp.status_code != 202:
            return TransformResult.error(
                {"reason": "api_error", "error_type": "submit_rejected", "status_code": resp.status_code}, retryable=False
            )
        operation_url = resp.headers["operation-location"] if "operation-location" in resp.headers else None
        if not operation_url:
            return TransformResult.error({"reason": "operation_location_missing"}, retryable=False)
        if not operation_location_host_matches(operation_url, self._endpoint):
            return TransformResult.error({"reason": "operation_location_untrusted"}, retryable=False)
        retry_after = self._parse_retry_after(resp.headers["retry-after"] if "retry-after" in resp.headers else None)
        return operation_url, retry_after

    def _poll(
        self,
        operation_url: str,
        state_id: str,
        *,
        token_id: str | None,
        retry_after: float | None,
        capacity_deadline: float,
        started_at: float,
    ) -> Mapping[str, Any] | TransformResult:
        client = self._get_http_client(state_id, token_id=token_id)
        poll_deadline = started_at + self._poll_timeout_seconds
        interval = min(
            retry_after if retry_after is not None else self._poll_interval_seconds,
            self._poll_max_interval_seconds,
        )

        while True:
            if self._shutdown.is_set():
                return self._shutdown_result(started_at)

            def do_call() -> Any:
                return client.get(operation_url, timeout=self._request_timeout_seconds)

            resp = self._http_call_with_capacity_retry(
                do_call, deadline=min(capacity_deadline, poll_deadline), started_at=started_at, error_type="poll_request_failed"
            )
            if isinstance(resp, TransformResult):
                return resp

            data = self._parse_json_strict(resp)
            if isinstance(data, TransformResult):
                return data
            status = data.get("status") if isinstance(data, dict) else None
            if status == "succeeded":
                analyze_result = data.get("analyzeResult")
                if not isinstance(analyze_result, dict):
                    return TransformResult.error({"reason": "malformed_response", "error_type": "missing_analyze_result"}, retryable=False)
                return analyze_result
            if status == "failed":
                error_obj = data.get("error")
                error_code = "unknown"
                if isinstance(error_obj, dict):
                    code = error_obj.get("code")
                    if isinstance(code, str):
                        error_code = code
                # Carry only the bounded Azure error.code token; never the raw error.message (Tier-3 egress).
                return TransformResult.error({"reason": "analysis_failed", "cause": error_code}, retryable=False)
            if status in ("notStarted", "running"):
                now = time.monotonic()
                if now >= poll_deadline:
                    return TransformResult.error({"reason": "poll_timeout", "elapsed_seconds": now - started_at}, retryable=False)
                sleep_seconds = min(interval, poll_deadline - now)
                if sleep_seconds > 0 and self._shutdown.wait(timeout=sleep_seconds):
                    return self._shutdown_result(started_at)
                interval = min(interval * self._poll_backoff_multiplier, self._poll_max_interval_seconds)
                continue
            return TransformResult.error({"reason": "malformed_response", "error_type": "unknown_status"}, retryable=False)

    def _parse_json_strict(self, response: Any) -> Any:
        text = response.text
        if type(text) is not str:
            return TransformResult.error({"reason": "malformed_response", "error_type": "non_text_body"}, retryable=False)
        try:
            data, parse_error = parse_json_strict(text)
        except RecursionError:
            return TransformResult.error({"reason": "malformed_response", "error_type": "json_too_deep"}, retryable=False)
        if parse_error is not None:
            return TransformResult.error({"reason": "malformed_response", "error_type": "invalid_json"}, retryable=False)
        return data

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        return seconds if seconds >= 0 else None

    def _shutdown_result(self, started_at: float) -> TransformResult:
        return TransformResult.error({"reason": "shutdown_requested", "elapsed_seconds": time.monotonic() - started_at}, retryable=False)

    def _retry_timeout_result(self, status_code: int, started_at: float) -> TransformResult:
        return TransformResult.error(
            {
                "reason": "retry_timeout",
                "status_code": status_code,
                "elapsed_seconds": time.monotonic() - started_at,
                "max_seconds": float(self._max_capacity_retry_seconds),
            },
            retryable=False,
        )

    def _build_enriched_result(self, row: PipelineRow, analyze_result: Mapping[str, Any], *, operation_url: str) -> TransformResult:
        try:
            output = row.to_dict()
            if self._content_field is not None:
                output[self._content_field] = extract_content(analyze_result)
            if self._page_count_field is not None:
                output[self._page_count_field] = count_pages(analyze_result)
            if self._result_field is not None:
                output[self._result_field] = dict(analyze_result)
            for azure_key, field_name in self._facet_fields.items():
                output[field_name] = extract_facet_list(analyze_result, azure_key)
        except MalformedResponseError as e:
            return TransformResult.error({"reason": "malformed_response", "message": str(e)}, retryable=False)

        output_contract = narrow_contract_to_output(input_contract=row.contract, output_row=output)
        output_contract = self._apply_declared_output_field_contracts(output_contract)
        output_contract = self._align_output_contract(output_contract)

        model_id_val = analyze_result.get("modelId")
        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "enriched",
                "fields_added": sorted(self.declared_output_fields),
                "metadata": {
                    "model_id": model_id_val if isinstance(model_id_val, str) else self._model_id,
                    "api_version": self._api_version,
                    "operation_id": operation_id_from_url(operation_url),
                    "page_count": count_pages(analyze_result),
                    "content_format": self._output_content_format,
                    "features": list(self._cfg.features),
                    "result_status": "succeeded",
                },
            },
        )

    # ── ADR-009 forward invariant probe ──────────────────────────────────

    def forward_invariant_probe_rows(self, probe: Any) -> list[Any]:
        return [self._augment_invariant_probe_row(probe, field_name=self._source_field, value="https://probe.example/doc.pdf")]

    def execute_forward_invariant_probe(self, probe_rows: list[Any], ctx: Any) -> TransformResult:
        """Exercise the real LRO path with a local fake client.

        Inserts the fake directly into ``_http_clients[state_id]`` so
        ``_get_http_client`` returns it without creating a real client — correctness
        depends on ``_get_http_client`` checking the cache before construction.
        """
        if len(probe_rows) != 1:
            raise FrameworkBugError(
                f"{self.__class__.__name__}.execute_forward_invariant_probe() requires exactly 1 row, got {len(probe_rows)}."
            )
        op_url = (
            f"{self._endpoint}/documentintelligence/documentModels/{self._model_id}/analyzeResults/probe?api-version={self._api_version}"
        )

        import json as _json

        class _ProbeResponse:
            def __init__(self, status_code: int, headers: dict[str, str], payload: dict[str, Any]) -> None:
                self.status_code = status_code
                self.headers = httpx.Headers(headers)
                self.text = _json.dumps(payload)
                self.content = self.text.encode()

            def raise_for_status(self) -> None:
                return None

        class _ProbeClient:
            def post(self, url: str, *, json: Any = None, timeout: float | None = None) -> Any:
                del url, json, timeout
                return _ProbeResponse(202, {"operation-location": op_url}, {})

            def get(self, url: str, *, timeout: float | None = None) -> Any:
                del url, timeout
                return _ProbeResponse(200, {}, {"status": "succeeded", "analyzeResult": {"content": "probe"}})

            def close(self) -> None:
                return None

        # op_url is built on the configured endpoint host, so the host-pin check passes.
        state_id = ctx.state_id or "invariant-probe-state"
        token_id = ctx.token.token_id if ctx.token is not None else None
        with self._http_clients_lock:
            self._http_clients[state_id] = _ProbeClient()
        try:
            return self._process_single_with_state(probe_rows[0], state_id, token_id=token_id)
        finally:
            with self._http_clients_lock:
                client = self._http_clients.pop(state_id, None)
            if client is not None:
                client.close()

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary=(
                    "Enrich rows with Azure Document Intelligence: submit a document URL or base64 string "
                    "and add extracted content (text/markdown) plus facets (pages, tables, key-value pairs, "
                    "paragraphs, typed model documents). Async analyze, fully audited."
                ),
                composer_hints=(
                    "source_mode chooses how the document is read: 'url' (a reachable/SAS URL in source_field) or 'base64'.",
                    "model_id selects extraction: prebuilt-read/layout/document, prebuilt-invoice/receipt/idDocument/tax, or a custom model id.",
                    "Configure at least one output: content_field and/or extract.{tables,key_value_pairs,documents,...}; each becomes a row field.",
                    "output_content_format: markdown gives section-structured content; text gives plain text.",
                    "Extraction is external and non-deterministic; route on_error to a quarantine sink. Malformed Azure responses fail closed.",
                ),
            )
        return None
