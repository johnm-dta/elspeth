# Azure Document Intelligence Transform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `azure_document_intelligence` transform — a row-enrichment plugin that submits a document reference to Azure AI Document Intelligence, awaits the async analysis, and enriches the row with extracted content + structured facets, fully audited.

**Architecture:** Self-contained `AzureDocumentIntelligence(BaseTransform, BatchTransformMixin)` (NOT subclassing the Azure safety base). All HTTP via `AuditedHTTPClient`. Async long-running-operation (LRO): `POST` → `202` + `Operation-Location` → poll `GET` until terminal status. Pure Tier-3 facet parsing in a sibling module. Enrichment idiom mirrors `web_scrape`.

**Tech Stack:** Python 3.12/3.13, Pydantic v2, httpx (via `AuditedHTTPClient`), pytest. No new third-party dependency (no Azure SDK).

Spec: `docs/superpowers/specs/2026-06-30-azure-document-intelligence-transform-design.md`.

## Global Constraints

- **Plugin name:** `azure_document_intelligence`. **determinism:** `Determinism.EXTERNAL_CALL`. **plugin_version:** `"1.0.0"`. **passes_through_input:** `True`. **creates_tokens:** `False`.
- **No new dependency.** Use `httpx` + `AuditedHTTPClient`, never `azure-ai-documentintelligence`.
- **API version:** `2024-11-30` only; `_SUPPORTED_API_VERSIONS = frozenset({"2024-11-30"})`.
- **Tier-3 discipline:** every external response via `parse_json_strict`; fail-closed `MalformedResponseError`; never fabricate absent fields. No raw Azure `error.message` in any `TransformResult.error` reason (only bounded `error.code` token). Closed error-reason vocabulary (Task 4).
- **Security:** endpoint HTTPS-validated; `api_key` `repr=False`, non-empty; `Operation-Location` host MUST match endpoint host before polling; base64 size-guarded.
- **Every declared output field is ALWAYS set** on success (empty container when the model produced nothing) — upholds `passes_through_input=True`.
- **Files live in** `src/elspeth/plugins/transforms/azure/`. Discovery is automatic (filesystem + `issubclass`); no registry edit.
- **Commits:** frequent, conventional, end every message with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Run from the worktree `/home/john/elspeth/.worktrees/azure-doc-intelligence` with `.venv/bin/python`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/elspeth/plugins/transforms/azure/document_intelligence.py` | `ExtractFields`, `AzureDocumentIntelligenceConfig`, `AzureDocumentIntelligence` transform (lifecycle, LRO transport, enrichment output, assistance). |
| `src/elspeth/plugins/transforms/azure/document_intelligence_result.py` | Pure Tier-3 helpers: facet/content/page-count extraction, `operation_location_host_matches`, `operation_id_from_url`. No I/O. |
| `tests/unit/plugins/transforms/azure/test_document_intelligence.py` | Config validation, LRO happy path, every error path, enrichment output, assistance. |
| `tests/unit/plugins/transforms/azure/test_document_intelligence_result.py` | Pure parser unit tests. |
| `tests/unit/contracts/transform_contracts/test_azure_document_intelligence_contract.py` | ADR-009 forward/backward invariant. |
| `tests/golden/web/catalog/knob_schema/transform__azure_document_intelligence.json` | Golden knob schema. |

---

## Task 1: Configuration models

**Files:**
- Create: `src/elspeth/plugins/transforms/azure/document_intelligence.py` (config portion only this task)
- Test: `tests/unit/plugins/transforms/azure/test_document_intelligence.py`

**Interfaces:**
- Produces: `ExtractFields` (frozen pydantic), `AzureDocumentIntelligenceConfig(TransformDataConfig)` with attributes used by later tasks: `endpoint, api_key, api_version, model_id, source_mode, source_field, max_base64_chars, content_field, output_content_format, extract, page_count_field, result_field, pages, locale, string_index_type, features, query_fields, poll_interval_seconds, poll_backoff_multiplier, poll_max_interval_seconds, poll_timeout_seconds, request_timeout_seconds, max_response_body_bytes, max_capacity_retry_seconds, batch_wait_timeout_seconds`; plus helper `configured_output_fields() -> dict[str, str]` mapping `azure_facet_key -> row_field` and `all_output_field_names() -> list[str]`.

- [ ] **Step 1: Write the failing test (valid config + key validation rules)**

```python
# tests/unit/plugins/transforms/azure/test_document_intelligence.py
from __future__ import annotations

import pytest

from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.azure.document_intelligence import (
    AzureDocumentIntelligenceConfig,
)

BASE = {
    "endpoint": "https://di.cognitiveservices.azure.com",
    "api_key": "k",
    "model_id": "prebuilt-layout",
    "source_mode": "url",
    "source_field": "doc_url",
    "content_field": "di_content",
    "schema": {"mode": "observed"},
}


def _cfg(**overrides):
    data = {**BASE, **overrides}
    return AzureDocumentIntelligenceConfig.from_dict(data, plugin_name="azure_document_intelligence")


def test_valid_minimal_config():
    cfg = _cfg()
    assert cfg.api_version == "2024-11-30"
    assert cfg.output_content_format == "text"
    assert cfg.configured_output_fields() == {}
    assert cfg.all_output_field_names() == ["di_content"]


def test_rejects_http_endpoint():
    with pytest.raises(PluginConfigError):
        _cfg(endpoint="http://di.cognitiveservices.azure.com")


def test_rejects_empty_api_key():
    with pytest.raises(PluginConfigError):
        _cfg(api_key="   ")


def test_rejects_unknown_api_version():
    with pytest.raises(PluginConfigError):
        _cfg(api_version="2023-07-31")


def test_rejects_bad_model_id():
    with pytest.raises(PluginConfigError):
        _cfg(model_id="bad/model id")


def test_requires_at_least_one_output():
    data = {k: v for k, v in BASE.items() if k != "content_field"}
    with pytest.raises(PluginConfigError):
        AzureDocumentIntelligenceConfig.from_dict(data, plugin_name="azure_document_intelligence")


def test_rejects_duplicate_output_field_names():
    with pytest.raises(PluginConfigError):
        _cfg(content_field="dup", extract={"tables": "dup"})


def test_query_fields_requires_feature():
    with pytest.raises(PluginConfigError):
        _cfg(query_fields=["Total"])


def test_query_fields_feature_requires_list():
    with pytest.raises(PluginConfigError):
        _cfg(features=["queryFields"])


def test_unknown_feature_rejected():
    with pytest.raises(PluginConfigError):
        _cfg(features=["nope"])


def test_base64_mode_and_extract_fields():
    cfg = _cfg(source_mode="base64", source_field="doc_b64",
               extract={"tables": "di_tables", "key_value_pairs": "di_kv"})
    assert cfg.configured_output_fields() == {"tables": "di_tables", "keyValuePairs": "di_kv"}
    assert set(cfg.all_output_field_names()) == {"di_content", "di_tables", "di_kv"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/azure/test_document_intelligence.py -q`
Expected: FAIL (`ImportError` / `ModuleNotFoundError`).

- [ ] **Step 3: Write the config models**

```python
# src/elspeth/plugins/transforms/azure/document_intelligence.py
"""Azure AI Document Intelligence enrichment transform.

Submits a document reference (URL or base64) per row to Azure Document
Intelligence, waits for the asynchronous analysis (POST 202 + Operation-Location
poll), and enriches the row with extracted content and structured facets.

All HTTP flows through AuditedHTTPClient (full request/response audit, header
fingerprinting, telemetry, rate limiting). GA api-version 2024-11-30. See
docs/superpowers/specs/2026-06-30-azure-document-intelligence-transform-design.md.

SECURITY: a SAS token embedded in a urlSource value is forwarded to Azure and
recorded in the audited request blob. Operator data; documented here.
"""

from __future__ import annotations

import re
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.url_validation import validate_credential_safe_https_url

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
    """Facet -> output-field-name map. Only set facets are emitted as row fields."""

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
    result_field: str | None = Field(None, description="Row field for the entire analyzeResult object (max fidelity).")

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
    max_capacity_retry_seconds: int = Field(3600, gt=0, description="Per-call 429/503 retry budget.")
    batch_wait_timeout_seconds: int = Field(3600, gt=0, description="Batch-mixin per-row wait timeout.")

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint(cls, v: str) -> str:
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
        unknown = [f for f in v if f not in _KNOWN_FEATURES]
        if unknown:
            raise ValueError(f"Unknown features {unknown}. Known: {sorted(_KNOWN_FEATURES)}.")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        names = self.all_output_field_names()
        if not names:
            raise ValueError(
                "At least one output target must be configured: content_field, page_count_field, "
                "result_field, or one entry in extract."
            )
        dupes = sorted({n for n in names if names.count(n) > 1})
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
        """Every output row field name this transform may add (order: content, page_count, result, facets)."""
        names: list[str] = []
        if self.content_field is not None:
            names.append(self.content_field)
        if self.page_count_field is not None:
            names.append(self.page_count_field)
        if self.result_field is not None:
            names.append(self.result_field)
        names.extend(self.configured_output_fields().values())
        return names
```

- [ ] **Step 4: Run config tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/azure/test_document_intelligence.py -q`
Expected: PASS (all config tests).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/transforms/azure/document_intelligence.py tests/unit/plugins/transforms/azure/test_document_intelligence.py
git commit -m "feat(plugins): azure_document_intelligence config models

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Pure Tier-3 result parser

**Files:**
- Create: `src/elspeth/plugins/transforms/azure/document_intelligence_result.py`
- Test: `tests/unit/plugins/transforms/azure/test_document_intelligence_result.py`

**Interfaces:**
- Consumes: `azure/errors.py::MalformedResponseError`.
- Produces: `extract_content(analyze_result) -> str`, `extract_facet_list(analyze_result, azure_key) -> list[Any]`, `count_pages(analyze_result) -> int`, `operation_location_host_matches(operation_url, endpoint) -> bool`, `operation_id_from_url(operation_url) -> str | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/plugins/transforms/azure/test_document_intelligence_result.py
from __future__ import annotations

import pytest

from elspeth.plugins.transforms.azure.document_intelligence_result import (
    count_pages,
    extract_content,
    extract_facet_list,
    operation_id_from_url,
    operation_location_host_matches,
)
from elspeth.plugins.transforms.azure.errors import MalformedResponseError


def test_extract_content_present_and_absent():
    assert extract_content({"content": "# Hi"}) == "# Hi"
    assert extract_content({}) == ""


def test_extract_content_wrong_type_fails_closed():
    with pytest.raises(MalformedResponseError):
        extract_content({"content": 123})


def test_extract_facet_list_present_absent_and_malformed():
    assert extract_facet_list({"tables": [{"rowCount": 1}]}, "tables") == [{"rowCount": 1}]
    assert extract_facet_list({}, "tables") == []
    with pytest.raises(MalformedResponseError):
        extract_facet_list({"tables": {"not": "a list"}}, "tables")


def test_count_pages():
    assert count_pages({"pages": [{}, {}, {}]}) == 3
    assert count_pages({}) == 0
    with pytest.raises(MalformedResponseError):
        count_pages({"pages": "x"})


def test_host_match():
    ep = "https://di.cognitiveservices.azure.com"
    assert operation_location_host_matches(
        "https://di.cognitiveservices.azure.com/documentintelligence/.../analyzeResults/abc?api-version=2024-11-30", ep
    )
    assert not operation_location_host_matches("https://evil.example.com/analyzeResults/abc", ep)
    assert not operation_location_host_matches("http://di.cognitiveservices.azure.com/x", ep)  # scheme mismatch
    assert not operation_location_host_matches("not a url", ep)


def test_operation_id_from_url():
    url = "https://h/documentintelligence/documentModels/prebuilt-layout/analyzeResults/3b31320d-8bab?api-version=2024-11-30"
    assert operation_id_from_url(url) == "3b31320d-8bab"
    assert operation_id_from_url("https://h/no/result/segment") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/azure/test_document_intelligence_result.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement the parser**

```python
# src/elspeth/plugins/transforms/azure/document_intelligence_result.py
"""Pure Tier-3 parsing helpers for Azure Document Intelligence analyzeResult.

No I/O. Every function fail-closes (MalformedResponseError) on a structurally
invalid result; absent optional facets return an empty container, never fabricated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from elspeth.plugins.transforms.azure.errors import MalformedResponseError


def extract_content(analyze_result: Mapping[str, Any]) -> str:
    """Return analyzeResult.content, or '' when absent. Fail closed on wrong type."""
    content = analyze_result.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise MalformedResponseError(f"analyzeResult.content must be str, got {type(content).__name__}")
    return content


def extract_facet_list(analyze_result: Mapping[str, Any], azure_key: str) -> list[Any]:
    """Return analyzeResult[azure_key] (a list), or [] when absent. Fail closed on wrong type."""
    value = analyze_result.get(azure_key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise MalformedResponseError(f"analyzeResult.{azure_key} must be a list, got {type(value).__name__}")
    return value


def count_pages(analyze_result: Mapping[str, Any]) -> int:
    """Return the number of analyzed pages, or 0 when absent. Fail closed on wrong type."""
    return len(extract_facet_list(analyze_result, "pages"))


def operation_location_host_matches(operation_url: str, endpoint: str) -> bool:
    """True iff operation_url is a well-formed HTTPS URL on the same host as endpoint.

    Guards against the polled Operation-Location (which carries our api-key) being
    pointed at an attacker host by a malformed/compromised 202 response.
    """
    try:
        op = urlparse(operation_url)
        ep = urlparse(endpoint)
    except ValueError:
        return False
    if op.scheme != "https" or not op.hostname:
        return False
    return op.hostname.lower() == (ep.hostname or "").lower()


def operation_id_from_url(operation_url: str) -> str | None:
    """Extract the result id from .../analyzeResults/{id}[?...], or None."""
    path = urlparse(operation_url).path
    segments = [s for s in path.split("/") if s]
    for i, seg in enumerate(segments):
        if seg == "analyzeResults" and i + 1 < len(segments):
            return segments[i + 1]
    return None
```

- [ ] **Step 4: Run parser tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/azure/test_document_intelligence_result.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/transforms/azure/document_intelligence_result.py tests/unit/plugins/transforms/azure/test_document_intelligence_result.py
git commit -m "feat(plugins): DI analyzeResult Tier-3 parser

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Transform skeleton — lifecycle, batch wiring, client cache

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/document_intelligence.py` (add the transform class)
- Test: `tests/unit/plugins/transforms/azure/test_document_intelligence.py` (add)

**Interfaces:**
- Consumes: `AzureDocumentIntelligenceConfig`; `BaseTransform`, `BatchTransformMixin`, `OutputPort`; `AuditedHTTPClient`; `create_schema_from_config`; `make_warn_telemetry_before_start`; `Determinism`, `AuditCharacteristic`, `PluginAuditWriter`, `LifecycleContext`, `TransformContext`, `PipelineRow`, `TransformResult`.
- Produces: class `AzureDocumentIntelligence` with `name="azure_document_intelligence"`, `declared_output_fields`, `_output_schema_config`, `_get_http_client`, lifecycle methods, `probe_config()`. Per-row methods added in Task 4.

This task mirrors `BaseAzureSafetyTransform` lifecycle/client-cache idioms verbatim (read `transforms/azure/base.py` lines 100-216, 256-282, 488-533). Use exactly these call shapes: `self.init_batch_processing(max_pending=..., output=output, name=self.name, max_workers=max_pending, batch_wait_timeout=self._effective_batch_wait_timeout_seconds)`, `self.accept_row(row, ctx, self._process_row)`, `self.shutdown_batch_processing()`. `AuditedHTTPClient(execution=self._recorder, state_id=state_id, run_id=self._run_id, telemetry_emit=self._telemetry_emit, timeout=self._request_timeout_seconds, headers=self._default_request_headers(), limiter=self._limiter, token_id=token_id, max_response_body_bytes=self._max_response_body_bytes)`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/plugins/transforms/azure/test_document_intelligence.py
from elspeth.contracts import Determinism
from elspeth.plugins.transforms.azure.document_intelligence import AzureDocumentIntelligence


def _transform(**overrides):
    data = {**BASE, **overrides}
    return AzureDocumentIntelligence(data)


def test_transform_metadata_and_declared_fields():
    t = _transform(extract={"tables": "di_tables"}, page_count_field="di_pages")
    assert t.name == "azure_document_intelligence"
    assert t.determinism is Determinism.EXTERNAL_CALL
    assert t.passes_through_input is True
    assert t.declared_output_fields == frozenset({"di_content", "di_tables", "di_pages"})


def test_probe_config_instantiates():
    cfg = AzureDocumentIntelligence.probe_config()
    AzureDocumentIntelligence(cfg)  # must not raise


def test_process_raises_use_accept():
    import pytest
    t = _transform()
    with pytest.raises(NotImplementedError):
        t.process(object(), object())  # type: ignore[arg-type]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/azure/test_document_intelligence.py -q -k "metadata or probe or process_raises"`
Expected: FAIL (`AttributeError`/`ImportError`).

- [ ] **Step 3: Add the transform skeleton**

Append to `document_intelligence.py` (new imports at top of file):

```python
import threading
import time
from collections.abc import Callable, Mapping

import structlog

from elspeth.contracts import Determinism
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.enums import AuditCharacteristic
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.infrastructure.telemetry import make_warn_telemetry_before_start

logger = structlog.get_logger(__name__)
_warn_telemetry_before_start = make_warn_telemetry_before_start(logger)

_HTTP_TIMEOUT_HEADROOM_SECONDS = 30.0


class AzureDocumentIntelligence(BaseTransform, BatchTransformMixin):
    """Enrich rows with Azure Document Intelligence extraction (async LRO)."""

    name = "azure_document_intelligence"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:pending"  # recomputed in Task 7 (placeholder must be a sha256: literal for the hash normalizer to match)
    config_model = AzureDocumentIntelligenceConfig
    passes_through_input = True
    creates_tokens = False
    discovery_secret_requirements: Mapping[str, tuple[str, ...]] = {
        "api_key": ("AZURE_DOCUMENT_INTELLIGENCE_KEY",),
    }
    audit_characteristics = frozenset({AuditCharacteristic.CREDENTIALS})
    capability_tags = ("azure", "document", "ocr", "enrichment", "http")

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
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
        """Auth headers for the audited client. Isolated so AAD bearer is a clean later add."""
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
        raise NotImplementedError(
            f"{self.__class__.__name__} uses row-level pipelining. Use accept() instead of process()."
        )

    def _get_http_client(self, state_id: str, *, token_id: str | None = None) -> Any:
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/azure/test_document_intelligence.py -q`
Expected: PASS (config + skeleton tests). Per-row tests come in Task 4.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(plugins): DI transform skeleton (lifecycle, batch, client cache)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: LRO processing + enrichment output (the core)

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/document_intelligence.py`
- Test: `tests/unit/plugins/transforms/azure/test_document_intelligence.py` (add LRO + error + enrichment tests with a fake client)

**Interfaces:**
- Consumes: `CapacityError`, `is_capacity_error`, `PluginRetryableError`, `FrameworkBugError` from `elspeth.contracts.errors`; `parse_json_strict` from `infrastructure/clients/json_utils`; `narrow_contract_to_output` from `elspeth.contracts.contract_propagation`; parser helpers from Task 2; `MalformedResponseError`; `httpx`.
- Produces: `_process_row`, `_process_single_with_state`, `_http_call_with_capacity_retry`, `_submit`, `_poll`, `_build_enriched_result`, error-result builders.

**CRITICAL — error categories (verified against `contracts/errors.py`):** `TransformResult.error(reason)` requires `reason["reason"] ∈ TransformErrorCategory` (a closed `Literal`); a Tier-1 audit-write guard crashes on an invalid value, and mypy rejects undeclared `TransformErrorReason` keys. This task therefore **first** adds four DI categories to the Literal, then uses ONLY declared keys. Mapping (this supersedes any inline sketch below where they differ):

| State | `reason` | extra declared keys |
|-------|----------|---------------------|
| source field absent | `missing_field` | `field` |
| non-string field | `non_string_field` | `field`, `actual_type` |
| bad urlSource | `invalid_input` | `field`, `error_type="invalid_document_url"` |
| base64 over cap | `invalid_input` | `field`, `error_type="base64_too_large"`, `message` |
| submit non-202 | `api_error` | `error_type="submit_rejected"`, `status_code` |
| no Operation-Location | `operation_location_missing` *(NEW)* | — |
| host ≠ endpoint | `operation_location_untrusted` *(NEW)* | — |
| poll non-2xx | `api_error` | `error_type="poll_request_failed"`, `status_code` |
| Azure failed | `analysis_failed` *(NEW)* | `cause`=Azure `error.code` (bounded; no raw message) |
| LRO timeout | `poll_timeout` *(NEW)* | `elapsed_seconds` |
| bad json/unknown status/missing analyzeResult | `malformed_response` | `error_type` |
| capacity exhausted | `retry_timeout` | `status_code`, `elapsed_seconds`, `max_seconds` |
| shutdown | `shutdown_requested` | `elapsed_seconds` |

`TransformResult` accessors (verified `contracts/results.py`): success → `result.status == "success"`, `result.row` (a `PipelineRow`); error → `result.status == "error"`, `result.reason` (the dict). There is NO `is_success`/`error_data`.

**Design notes (nested budgets):** capacity-retry wraps each *individual* POST/GET (inner, bounded by `max_capacity_retry_seconds`); the poll loop is bounded by `poll_timeout_seconds` (outer). Poll-GET capacity retries use `min(capacity_deadline, poll_deadline)` so the inner loop never outlives the outer. A 429 on submit/poll feeds the inner capacity loop. `self._shutdown` interrupts both sleeps. The poll loop terminates on terminal `status` (data-driven) for replay parity. Network errors raise `PluginRetryableError` (engine retries the row); structural failures return `TransformResult.error(...)`.

- [ ] **Step 0: Extend the error-category Literal**

In `src/elspeth/contracts/errors.py`, append to the `TransformErrorCategory` Literal a DI section:

```python
    # Azure Document Intelligence (async analyze LRO)
    "analysis_failed",               # Azure analyze operation reported status=failed
    "poll_timeout",                  # async operation did not reach terminal status within budget
    "operation_location_missing",    # 202 response lacked the Operation-Location header
    "operation_location_untrusted",  # Operation-Location host != configured endpoint (security)
```

Add a tiny test asserting these are valid (e.g. construct `TransformResult.error({"reason": "poll_timeout"})` and the three others without raising). Run the existing `tests/unit/core/landscape/test_data_flow_repository.py` Tier-1 category-guard tests to confirm no regression.

- [ ] **Step 1: Write the failing tests (fake-client LRO happy path, errors, enrichment)**

```python
# add to tests/unit/plugins/transforms/azure/test_document_intelligence.py
import json
import httpx
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract


class _Resp:
    def __init__(self, status_code, *, headers=None, body=None):
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)
        self.content = self.text.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("GET", "https://x"),
                                        response=httpx.Response(self.status_code, request=httpx.Request("GET", "https://x")))


class _FakeClient:
    """Scripted POST then sequence of GET responses."""
    def __init__(self, post_resp, get_resps):
        self._post_resp = post_resp
        self._get_resps = list(get_resps)
        self.closed = False

    def post(self, url, *, json=None, timeout=None):
        return self._post_resp

    def get(self, url, *, timeout=None):
        return self._get_resps.pop(0)

    def close(self):
        self.closed = True


def _row(url="https://x/y.pdf"):
    contract = SchemaContract(mode="OBSERVED", fields=(
        FieldContract(normalized_name="doc_url", original_name="doc_url", python_type=str,
                      required=False, source="inferred", nullable=False),), locked=True)
    return PipelineRow({"doc_url": url}, contract)


def _run_with_fake(t, fake, row=None):
    row = row or _row()
    # poll fast in tests
    object.__setattr__  # no-op import guard
    t._poll_interval_seconds = 0.0
    t._poll_max_interval_seconds = 0.0
    with t._http_clients_lock:
        t._http_clients["s1"] = fake
    try:
        return t._process_single_with_state(row, "s1", token_id=None)
    finally:
        with t._http_clients_lock:
            t._http_clients.pop("s1", None)


OP = "https://test.cognitiveservices.azure.com/documentintelligence/documentModels/prebuilt-layout/analyzeResults/abc?api-version=2024-11-30"


def _t_for_lro(**overrides):
    data = {**BASE, "endpoint": "https://test.cognitiveservices.azure.com", **overrides}
    return AzureDocumentIntelligence(data)


def test_lro_happy_path_enriches():
    t = _t_for_lro(extract={"tables": "di_tables"})
    post = _Resp(202, headers={"operation-location": OP})
    running = _Resp(200, body={"status": "running"})
    done = _Resp(200, body={"status": "succeeded",
                            "analyzeResult": {"content": "# Doc", "tables": [{"rowCount": 2}], "pages": [{}, {}]}})
    result = _run_with_fake(t, _FakeClient(post, [running, done]))
    assert result.status == "success"
    out = result.row.to_dict()
    assert out["di_content"] == "# Doc"
    assert out["di_tables"] == [{"rowCount": 2}]
    assert out["doc_url"] == "https://x/y.pdf"


def test_facet_absent_emits_empty_container():
    t = _t_for_lro(extract={"tables": "di_tables"})
    post = _Resp(202, headers={"operation-location": OP})
    done = _Resp(200, body={"status": "succeeded", "analyzeResult": {"content": "x"}})
    result = _run_with_fake(t, _FakeClient(post, [done]))
    assert result.status == "success"
    assert result.row.to_dict()["di_tables"] == []  # always set


def test_missing_operation_location():
    t = _t_for_lro()
    result = _run_with_fake(t, _FakeClient(_Resp(202, headers={}), []))
    assert result.status == "error"
    assert result.reason["reason"] == "operation_location_missing"


def test_operation_location_host_mismatch():
    t = _t_for_lro()
    bad = "https://evil.example.com/x/analyzeResults/abc"
    result = _run_with_fake(t, _FakeClient(_Resp(202, headers={"operation-location": bad}), []))
    assert result.reason["reason"] == "operation_location_untrusted"


def test_analysis_failed_no_raw_message():
    t = _t_for_lro()
    post = _Resp(202, headers={"operation-location": OP})
    failed = _Resp(200, body={"status": "failed",
                              "error": {"code": "InvalidContent", "message": "secret detail leak"}})
    result = _run_with_fake(t, _FakeClient(post, [failed]))
    assert result.reason["reason"] == "analysis_failed"
    assert "secret detail leak" not in json.dumps(result.reason)
    assert result.reason.get("cause") == "InvalidContent"


def test_poll_timeout():
    t = _t_for_lro()
    t._poll_timeout_seconds = -1.0  # force immediate timeout
    post = _Resp(202, headers={"operation-location": OP})
    result = _run_with_fake(t, _FakeClient(post, [_Resp(200, body={"status": "running"})]))
    assert result.reason["reason"] == "poll_timeout"


def test_missing_source_field():
    t = _t_for_lro()
    empty = PipelineRow({}, _row().contract)
    result = _run_with_fake(t, _FakeClient(_Resp(202), []), row=empty)
    assert result.reason["reason"] == "missing_field"


def test_base64_too_large():
    t = _t_for_lro(source_mode="base64", source_field="doc_b64", max_base64_chars=4)
    contract = _row().contract
    big = PipelineRow({"doc_b64": "AAAAAAAA"}, contract)
    result = _run_with_fake(t, _FakeClient(_Resp(202), []), row=big)
    assert result.reason["reason"] == "base64_too_large"
```

> Note: confirm `TransformResult` accessor names (`is_success`, `row`, `error_data`) against `infrastructure/results.py` during implementation and adjust the test accessors to the real API before Step 4.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/azure/test_document_intelligence.py -q -k lro or facet or operation or analysis or poll or source_field or base64`
Expected: FAIL (`AttributeError: _process_single_with_state`).

- [ ] **Step 3: Implement per-row LRO processing**

Add to the class (full code):

```python
    _CAPACITY_RETRY_INITIAL_DELAY_SECONDS = 0.05
    _CAPACITY_RETRY_MAX_DELAY_SECONDS = 1.0

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
        if self._source_field not in row:
            return TransformResult.error({"reason": "missing_field", "field": self._source_field}, retryable=False)
        ref = row[self._source_field]
        if not isinstance(ref, str) or not ref.strip():
            return TransformResult.error(
                {"reason": "non_string_field", "field": self._source_field, "actual_type": type(ref).__name__},
                retryable=False,
            )
        if self._source_mode == "url":
            from urllib.parse import urlparse

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

        analyze = self._poll(operation_url, state_id, token_id=token_id, retry_after=retry_after,
                             capacity_deadline=capacity_deadline, started_at=started_at)
        if isinstance(analyze, TransformResult):
            return analyze

        return self._build_enriched_result(row, analyze, operation_url=operation_url)

    def _analyze_url(self) -> str:
        params = [("_overload", "analyzeDocument"), ("api-version", self._api_version)]
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
        from urllib.parse import urlencode

        query = urlencode(params)
        return f"{self._endpoint}/documentintelligence/documentModels/{self._model_id}:analyze?{query}"

    def _http_call_with_capacity_retry(
        self, do_call: Callable[[], Any], *, deadline: float, started_at: float, error_type: str,
    ) -> Any:
        """Run one HTTP call with capacity (429/503) retry. Returns httpx.Response or TransformResult.

        Non-capacity HTTP errors map to the `api_error` category with `error_type`
        as the sub-label. Raises PluginRetryableError on network errors (engine retries the row).
        """
        delay = self._CAPACITY_RETRY_INITIAL_DELAY_SECONDS
        while True:
            if self._shutdown.is_set():
                return self._shutdown_result(started_at)
            try:
                response = do_call()
                response.raise_for_status()
                return response
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

    def _submit(self, body, state_id, *, token_id, capacity_deadline, started_at):
        from elspeth.plugins.transforms.azure.document_intelligence_result import operation_location_host_matches

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

    def _poll(self, operation_url, state_id, *, token_id, retry_after, capacity_deadline, started_at):
        client = self._get_http_client(state_id, token_id=token_id)
        poll_deadline = started_at + self._poll_timeout_seconds
        interval = min(retry_after if retry_after is not None else self._poll_interval_seconds, self._poll_max_interval_seconds)

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
                error_code = error_obj.get("code") if isinstance(error_obj, dict) and isinstance(error_obj.get("code"), str) else "unknown"
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
        return TransformResult.error(
            {"reason": "shutdown_requested", "elapsed_seconds": time.monotonic() - started_at}, retryable=False
        )

    def _retry_timeout_result(self, status_code: int, started_at: float) -> TransformResult:
        return TransformResult.error(
            {"reason": "retry_timeout", "status_code": status_code, "elapsed_seconds": time.monotonic() - started_at,
             "max_seconds": self._max_capacity_retry_seconds},
            retryable=False,
        )

    def _build_enriched_result(self, row: PipelineRow, analyze_result: Mapping[str, Any], *, operation_url: str) -> TransformResult:
        from elspeth.contracts.contract_propagation import narrow_contract_to_output
        from elspeth.plugins.transforms.azure.document_intelligence_result import (
            count_pages,
            extract_content,
            extract_facet_list,
            operation_id_from_url,
        )

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
```

Add imports at file top:

```python
import httpx
from elspeth.contracts.errors import FrameworkBugError, PluginRetryableError, is_capacity_error
from elspeth.plugins.infrastructure.clients.json_utils import parse_json_strict
from elspeth.plugins.transforms.azure.errors import MalformedResponseError
```

- [ ] **Step 4: Run to verify pass** (after reconciling `TransformResult` accessor names)

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/azure/test_document_intelligence.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(plugins): DI LRO processing + enrichment output

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: ADR-009 forward/backward invariant probe

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/document_intelligence.py` (add probe hooks)
- Create: `tests/unit/contracts/transform_contracts/test_azure_document_intelligence_contract.py`

**Interfaces:**
- Consumes: `BaseTransform.execute_forward_invariant_probe` contract; `_augment_invariant_probe_row`.
- Produces: `forward_invariant_probe_rows`, `execute_forward_invariant_probe` (drives the real path with a local fake client that returns 202 + Operation-Location then a succeeded result).

Mirror `AzureContentSafety.execute_forward_invariant_probe` (read `transforms/azure/content_safety.py:152-206`), but the fake client must implement `post(url, *, json=None, timeout=None)` returning a 202 with an `operation-location` header on the probe endpoint host, and `get(url, *, timeout=None)` returning a succeeded body. Because the safety base's `_execute_forward_invariant_probe_with_client` is not inherited here, implement an equivalent: temporarily install the fake into `self._http_clients[state_id]`, call `_process_single_with_state`, then clean up.

- [ ] **Step 1: Write the failing contract test**

```python
# tests/unit/contracts/transform_contracts/test_azure_document_intelligence_contract.py
from __future__ import annotations

from elspeth.plugins.transforms.azure.document_intelligence import AzureDocumentIntelligence


def test_forward_invariant_probe_passes_through_and_enriches():
    t = AzureDocumentIntelligence(AzureDocumentIntelligence.probe_config())
    # The ADR-009 harness calls forward_invariant_probe_rows(probe) then
    # execute_forward_invariant_probe(rows, ctx). Mirror the safety contract test's
    # construction of probe/ctx (see test_azure_prompt_shield_contract.py) and assert
    # the result is success and carries content_field.
    ...
```

Implementation note: read `tests/unit/contracts/transform_contracts/test_azure_prompt_shield_contract.py` and `_azure_batch_helpers.py` and copy the probe/ctx construction exactly; replace the safety assertions with: result is success and `content_field` present.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/contracts/transform_contracts/test_azure_document_intelligence_contract.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement probe hooks**

```python
    def forward_invariant_probe_rows(self, probe: Any) -> list[Any]:
        return [self._augment_invariant_probe_row(probe, field_name=self._source_field, value="https://probe.example/doc.pdf")]

    def execute_forward_invariant_probe(self, probe_rows: list[Any], ctx: Any) -> TransformResult:
        if len(probe_rows) != 1:
            raise FrameworkBugError(
                f"{self.__class__.__name__}.execute_forward_invariant_probe() requires exactly 1 row, got {len(probe_rows)}."
            )
        op_url = f"{self._endpoint}/documentintelligence/documentModels/{self._model_id}/analyzeResults/probe?api-version={self._api_version}"

        class _ProbeResponse:
            def __init__(self, status_code: int, headers: dict[str, str], body: dict[str, Any]) -> None:
                self.status_code = status_code
                self.headers = httpx.Headers(headers)
                self.text = __import__("json").dumps(body)
                self.content = self.text.encode()

            def raise_for_status(self) -> None:
                return None

        class _ProbeClient:
            def post(self, url: str, *, json: Any = None, timeout: float | None = None) -> Any:
                return _ProbeResponse(202, {"operation-location": op_url}, {})

            def get(self, url: str, *, timeout: float | None = None) -> Any:
                return _ProbeResponse(200, {}, {"status": "succeeded", "analyzeResult": {"content": "probe"}})

            def close(self) -> None:
                return None

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
```

- [ ] **Step 4: Run the contract test + the ADR-009 invariant suite**

Run: `.venv/bin/python -m pytest tests/unit/contracts/transform_contracts/test_azure_document_intelligence_contract.py -q`
Also run any global invariant harness that enumerates `passes_through_input` transforms (search `tests/` for the skip-rate budget test) to confirm the new plugin is covered, not skipped.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(plugins): DI ADR-009 forward invariant probe

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Catalog assistance metadata + golden knob schema

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/document_intelligence.py` (assistance + usage prose)
- Create: `tests/golden/web/catalog/knob_schema/transform__azure_document_intelligence.json`

**Interfaces:**
- Produces: `get_agent_assistance`, `usage_when_to_use`, `usage_when_not_to_use`, `example_use`.

- [ ] **Step 1: Add assistance + usage prose**

```python
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
```

- [ ] **Step 2: Generate the golden knob schema**

The golden test `tests/unit/web/catalog/test_knob_schema_golden.py` builds each plugin's payload from the live catalog and (line 40-41) asserts the golden directory's file set equals the live set — so a new plugin REQUIRES a matching golden file. Generate it with the exact same lowering the test uses:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.service import CatalogServiceImpl

svc = CatalogServiceImpl(get_shared_plugin_manager())
info = svc._schema_cache[("transform", "azure_document_intelligence")]
payload = {"plugin_kind": "transform", "plugin_name": "azure_document_intelligence", "knob_schema": info.knob_schema}
out = Path("tests/golden/web/catalog/knob_schema/transform__azure_document_intelligence.json")
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("wrote", out)
PY
```

Then run the golden test:
`.venv/bin/python -m pytest tests/unit/web/catalog/test_knob_schema_golden.py -q`
Expected: PASS (golden present and matching the live catalog).

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat(plugins): DI catalog assistance + golden knob schema

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Finalize — source_file_hash, full gates, review

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/document_intelligence.py` (set `source_file_hash`)

- [ ] **Step 1: Compute and set `source_file_hash`**

`scripts/cicd/plugin_hash.py` is a library (no CLI). The placeholder line must already be a `"sha256:..."` literal (Task 3 set `"sha256:pending"`) so the normalizer matches it. Compute + rewrite in place:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash

p = Path("src/elspeth/plugins/transforms/azure/document_intelligence.py")
h = compute_source_file_hash(p)
fix_source_file_hash(p, "AzureDocumentIntelligence", h)
print("set source_file_hash to", h)
PY
```

Then run the plugin-hash CI test to confirm green (find it: `grep -rl plugin_hash tests/`; it is the test that imports `compute_source_file_hash` and asserts each plugin's declared hash matches). Run that test module with `.venv/bin/python -m pytest <that_module> -q`.

- [ ] **Step 2: Full local gates**

```bash
.venv/bin/ruff check src/elspeth/plugins/transforms/azure/document_intelligence.py src/elspeth/plugins/transforms/azure/document_intelligence_result.py tests/unit/plugins/transforms/azure/test_document_intelligence.py tests/unit/plugins/transforms/azure/test_document_intelligence_result.py
.venv/bin/mypy src/elspeth/plugins/transforms/azure/document_intelligence.py src/elspeth/plugins/transforms/azure/document_intelligence_result.py
.venv/bin/python -m pytest tests/unit/plugins/transforms/azure tests/unit/contracts/transform_contracts -q
wardline scan . --fail-on ERROR
```

Expected: ruff clean, mypy clean, tests pass, wardline exit 0 (fix boundary findings if any).

- [ ] **Step 3: Broader regression slice**

```bash
.venv/bin/python -m pytest tests/unit/plugins tests/unit/contracts tests/golden/web/catalog -q
```

Expected: PASS (no regressions; the discovery scan now includes the new plugin).

- [ ] **Step 4: Code-review cycle**

Dispatch `pr-review-toolkit:code-reviewer` and `pr-review-toolkit:silent-failure-hunter` over the diff; address findings; re-run gates. Repeat until clean.

- [ ] **Step 5: Final commit + pause for merge**

```bash
git add -A && git commit -m "feat(plugins): finalize azure_document_intelligence (hash, gates)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Then PAUSE and surface to the operator for the release-branch merge decision (do not merge to `release/0.7.0` autonomously).

---

## Self-Review

**Spec coverage:** §3 architecture → Task 3/4 (self-contained class). §4 config → Task 1 (every field + every §4.7 rule). §5 execution/LRO → Task 4 (submit/poll/budgets/shutdown/replay-data-driven). §6 output → Task 4 `_build_enriched_result`. §7 security → Task 1 (endpoint/api_key) + Task 2 (host-pin) + Task 4 (no-raw-message, base64 guard) + closed reason vocab. §8 audit → AuditedHTTPClient (Task 3) + success_reason metadata (Task 4). §9 compatibility → Task 1 (api_version allowlist, model_id, features) + Task 4 (`_analyze_url`). §10 testing → Tasks 1,2,4,5,6. §12 files → all tasks.

**Placeholder scan:** Two deliberate "locate the command" steps remain — golden-knob-schema generator (Task 6) and `plugin_hash` tool (Task 7) — because their exact invocation is environment-specific; both name the search target and the verifying command. The contract-test body (Task 5 Step 1) is a `...` stub by design: it must copy the exact probe/ctx construction from the existing safety contract test, which the implementer reads in-repo. All production code is complete.

**Type consistency:** `configured_output_fields()` (azure_key→field) and `all_output_field_names()` are used consistently across Tasks 1/3/4. `_process_single_with_state(row, state_id, *, token_id)` signature is identical in Tasks 4 and 5. `_http_call_with_capacity_retry(do_call, *, deadline, started_at, error_reason)` consistent. Reason vocabulary fixed in Task 4 and matches the spec (spec to add `poll_request_failed`).

**Open verification items for the implementer (not blockers):**
1. Golden-knob-schema generator command (Task 6) — locate the catalog golden-regen target.
2. `plugin_hash` tool path (Task 7) — `scripts/cicd/plugin_hash*`.
3. Contract-test probe/ctx construction (Task 5 Step 1) — copy from `test_azure_prompt_shield_contract.py`.

**Resolved during planning (verified against source):** `TransformResult` accessors are `.status` / `.row` / `.reason` (not `is_success`/`error_data`); error `reason` values must be valid `TransformErrorCategory` members → Step 0 adds four DI categories; `TransformErrorReason` extra keys must be declared (`error_type`, `cause`, `message`, `status_code`, `field`, `actual_type`, `elapsed_seconds`, `max_seconds`); `AuditCharacteristic.CREDENTIALS` and `PluginAssistance(plugin_name, issue_code, summary, composer_hints)` confirmed; PipelineRow accepts nested dict/list facet values (round-trip tested).
