# Plugin Expansion Design — Search, Analytical, and Reporting Tools

**Date**: 2026-04-14
**Status**: Draft
**Approach**: Hybrid — provider registry for search backends, individual plugins for scraping/reporting

## Context

ELSPETH's plugin system currently covers data processing (CSV/JSON/Dataverse sources/sinks), LLM transforms (Azure OpenAI, OpenRouter), RAG retrieval (Azure Search, Chroma), content safety (Azure), and basic web scraping (HTTP + BeautifulSoup). This design expands the plugin ecosystem to support six new capability areas:

1. **Web scraping and data extraction** — JS-rendered pages, interactive automation
2. **Real-time market and trend analysis** — monitoring, change detection, aggregation
3. **Content marketing** — multi-source research, structured summarization
4. **Automated research and reporting** — multi-step investigation, report generation
5. **Interactive web automation** — browser-based data extraction, form interaction
6. **Enterprise knowledge management** — document ingestion, indexing, semantic search

The near-term validation target is a **web research pipeline**: scrape sites (including JS-rendered) → LLM extraction/summarization → index into search backend → query for analysis → generate reports.

## Architecture Decision: Hybrid Approach

**Search backends** get a provider registry (like the LLM transform's `_PROVIDERS` dict). One `search_source`, one `search_sink`, one RAG provider — each with pluggable backends. Search backends genuinely share an interface (index documents, query documents, delete documents), so the abstraction pays for itself.

**Web scraping** extends the existing `web_scrape` transform with a `mode` field (`"http"` or `"browser"`). Same output contract, same security infrastructure, two execution paths.

**Reporting** is an individual `report_sink` plugin. Reports are too varied (and terminal) to need a provider registry.

**Notifications** are a separate `notification_sink` plugin (email, Slack, Teams, webhooks).

### Alternatives Considered

- **Approach A (Unified Platform)**: Provider registries for everything — search, scraping, and reporting. Rejected because scraping and reporting don't have multiple backends that share a meaningful interface. Over-abstraction.
- **Approach B (Individual Plugins)**: No shared abstractions. Each search backend is its own independent source/sink/RAG. Rejected because search backends genuinely share an interface, and duplicating source/sink/RAG logic per backend doesn't scale to 3-4 backends.

---

## 1. Search Plugin Architecture

### 1.1 Search Protocol

All search backends implement the `SearchProvider` protocol (structural typing, same pattern as `LLMProvider`):

```python
@runtime_checkable
class SearchProvider(Protocol):
    @property
    def capabilities(self) -> SearchCapabilities: ...

    def index_documents(
        self, documents: Sequence[SearchDocument], *, index: str
    ) -> IndexResult: ...

    def search(
        self, query: SearchQuery, *, index: str
    ) -> SearchResult: ...

    def delete_documents(
        self, ids: Sequence[str], *, index: str
    ) -> DeleteResult: ...

    def ensure_index(
        self, index: str, schema: IndexSchema | None = None
    ) -> None: ...

    def close(self) -> None: ...
```

### 1.2 Core Data Types

All frozen dataclasses with `deep_freeze` contracts per ELSPETH conventions.

```python
@dataclass(frozen=True, slots=True)
class IndexResult:
    indexed_count: int
    failed_count: int
    failed_ids: Sequence[str]                  # IDs that failed to index

@dataclass(frozen=True, slots=True)
class DeleteResult:
    deleted_count: int
    not_found_ids: Sequence[str]

@dataclass(frozen=True, slots=True)
class IndexSchema:
    """Declarative index schema for ensure_index()."""
    fields: Mapping[str, IndexFieldType]       # field_name → type
    vector_dimensions: int | None = None       # Required if vector search enabled
    vector_metric: Literal["cosine", "l2", "dot"] = "cosine"

@dataclass(frozen=True, slots=True)
class HighlightConfig:
    fields: Sequence[str] | None = None        # None = all text fields
    pre_tag: str = "<em>"
    post_tag: str = "</em>"
    fragment_size: int = 150                   # Characters per highlight snippet
    max_fragments: int = 3

@dataclass(frozen=True, slots=True)
class SearchDocument:
    id: str                                    # Document ID
    content: str                               # Primary text content
    metadata: Mapping[str, Any]                # Filterable metadata fields
    embedding: Sequence[float] | None = None   # Optional pre-computed vector

@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    content: str
    score: float
    metadata: Mapping[str, Any]
    highlights: Mapping[str, Sequence[str]] | None = None

@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: Sequence[SearchHit]
    total: int
    next_cursor: str | None = None             # None = no more results
    aggregations: Mapping[str, Any] | None = None
    facets: Mapping[str, Sequence[FacetCount]] | None = None

@dataclass(frozen=True, slots=True)
class FacetCount:
    value: str
    count: int
```

### 1.3 Query Model

The query model supports core retrieval (all providers), structured filtering (all providers), result shaping (optional), and analytics (optional):

```python
@dataclass(frozen=True, slots=True)
class SearchQuery:
    # Core (all providers MUST support)
    text: str | None = None
    vector: Sequence[float] | None = None
    mode: Literal["text", "vector", "hybrid"] = "hybrid"
    top_k: int = 10
    offset: int = 0
    cursor: str | None = None                  # Opaque cursor from previous result

    # Structured filtering (all providers MUST support)
    filters: FilterGroup | None = None

    # Result shaping (optional per provider)
    sort: Sequence[SortSpec] | None = None
    highlight: HighlightConfig | None = None

    # Analytics (optional per provider)
    aggregations: Mapping[str, AggregationSpec] | None = None
    facets: Sequence[str] | None = None

@dataclass(frozen=True, slots=True)
class FilterGroup:
    logic: Literal["and", "or", "not"] = "and"
    conditions: Sequence[FilterCondition | FilterGroup]   # Recursive nesting

@dataclass(frozen=True, slots=True)
class FilterCondition:
    field: str
    operator: Literal["eq", "neq", "gt", "gte", "lt", "lte",
                       "in", "not_in", "exists", "range", "prefix"]
    value: Any

@dataclass(frozen=True, slots=True)
class SortSpec:
    field: str | Literal["_score", "_id"]
    order: Literal["asc", "desc"] = "desc"

@dataclass(frozen=True, slots=True)
class AggregationSpec:
    type: Literal["terms", "histogram", "date_histogram",
                   "avg", "sum", "min", "max", "count", "cardinality"]
    field: str
    options: Mapping[str, Any] | None = None
```

### 1.4 Provider Capability Declaration

Each provider declares its capabilities. Validation at config time rejects unsupported features.

```python
@dataclass(frozen=True, slots=True)
class SearchCapabilities:
    supports_vector: bool
    supports_hybrid: bool
    supports_aggregations: bool
    supports_facets: bool
    supports_highlighting: bool
    supports_nested_filters: bool
    supports_cursor_pagination: bool
    max_top_k: int | None = None
```

**Capability matrix (initial providers):**

| Capability | OpenSearch | Qdrant | Meilisearch | Chroma |
|-----------|-----------|--------|-------------|--------|
| Full-text (BM25) | Yes | Basic | Yes | No |
| Vector search | Yes (k-NN) | Yes | Yes | Yes |
| Hybrid | Yes | Yes (recent) | Yes | No |
| Aggregations | Yes (full) | No | No | No |
| Facets | Yes | No | Yes | No |
| Highlighting | Yes (3 modes) | No | Yes | No |
| Nested filters | Yes | Yes | Partial | Partial |
| Cursor pagination | Yes (PIT) | Yes (scroll) | No | No |

### 1.5 Provider Registry

Same pattern as LLM transform:

```python
_SEARCH_PROVIDERS: dict[str, tuple[type[ProviderConfig], type[SearchProvider]]] = {
    "opensearch": (OpenSearchConfig, OpenSearchProvider),
    "qdrant": (QdrantConfig, QdrantProvider),
    "meilisearch": (MeilisearchConfig, MeilisearchProvider),
    "chroma": (ChromaSearchConfig, ChromaSearchProvider),
}
```

### 1.6 Three Plugins Using the Registry

| Plugin | Type | Role | Config Highlights |
|--------|------|------|-------------------|
| `search_source` | Source | Query search index as pipeline input | provider, index, query template (Jinja2 — receives pipeline metadata as context), top_k, cursor iteration |
| `search_sink` | Sink | Index pipeline output | provider, index, field mapping (content/id/metadata), on_duplicate, batch_size |
| Search RAG provider | RAG provider | Added to existing RAG `_PROVIDER_REGISTRY` | provider, index — registered alongside Azure Search and Chroma |

**search_source query templating**: The Jinja2 query template receives `{metadata: {...}, params: {...}}` as context (pipeline metadata and runtime parameters). This follows the same pattern as the LLM transform's `template` field. Static queries (no template) are also supported for index-scan use cases.

**Chroma upgrade path**: The existing `chroma_sink.py` stays as-is. A new `ChromaSearchProvider` implements the `SearchProvider` protocol, wrapping Chroma's query API. The Chroma sink remains independently usable — the search provider is an additional integration point, not a replacement.

### 1.7 Tier 3 Boundary Handling

All search operations sit at Tier 3 boundaries:

- **Index operations (sink)**: Wrap via `AuditedHTTPClient`. Record indexed documents. Divert failed documents via `_divert_row()`.
- **Search operations (source/RAG)**: Wrap via `AuditedHTTPClient`. Record query + response. Validate response structure (fail closed on unexpected shapes). Coerce at boundary.
- **Connection failures**: Typed exceptions — retryable for network/5xx/429, non-retryable for auth/4xx. Plugged into engine `RetryManager`.

### 1.8 Search Backend Evaluation

**Build order: OpenSearch → Qdrant → Meilisearch** (plus Chroma upgrade from existing sink).

| Factor | OpenSearch | Qdrant | Meilisearch |
|--------|-----------|--------|-------------|
| **License** | Apache 2.0 | Apache 2.0 | MIT |
| **Primary strength** | Analytics + hybrid search | Vector search + filtering | Developer UX, lightweight |
| **Aggregations** | Full (bucket, metric, pipeline) | No | No |
| **Weight** | Heavy (JVM, 1-4GB) | Medium (Rust, 200-500MB) | Light (Rust, 50-100MB) |
| **Python SDK** | opensearch-py | qdrant-client | meilisearch |
| **Cloud managed** | AWS OpenSearch, Aiven | Qdrant Cloud | Meilisearch Cloud |

**Rationale**: OpenSearch first because aggregations are essential for analytical/reporting workflows and it exercises the full capability surface of the protocol. If the abstraction handles OpenSearch, it handles everything simpler. Qdrant second for optimized vector RAG. Meilisearch third for lightweight development.

---

## 2. Enhanced Web Scraping

### 2.1 Mode-Based Dispatch

Extend the existing `web_scrape` transform with a `mode` field:

```python
class WebScrapeConfig(TransformDataConfig):
    # Existing fields unchanged
    url_field: str
    content_field: str
    fingerprint_field: str
    format: Literal["markdown", "text", "raw"] = "markdown"
    fingerprint_mode: Literal["content", "full"] = "content"
    http: WebScrapeHTTPConfig

    # New fields
    mode: Literal["http", "browser"] = "http"
    browser: BrowserConfig | None = None     # Required when mode="browser"
```

### 2.2 Browser Configuration

```python
class BrowserConfig(BaseModel):
    headless: bool = True
    wait_for: Literal["load", "domcontentloaded", "networkidle"] = "networkidle"
    wait_timeout_ms: int = Field(default=30_000, ge=1000, le=120_000)
    viewport: ViewportConfig | None = None
    javascript_enabled: bool = True

    # Declarative interactions
    interactions: Sequence[InteractionStep] | None = None

    # Resource blocking (performance + security)
    block_resources: Sequence[Literal[
        "image", "media", "font", "stylesheet", "websocket"
    ]] = ("image", "media", "font")

    # Visual capture
    capture: CaptureConfig | None = None

class InteractionStep(BaseModel):
    action: Literal["click", "scroll", "fill", "wait", "screenshot"]
    selector: str | None = None
    value: str | None = None
    wait_ms: int = Field(default=1000, ge=100, le=30_000)

class ViewportConfig(BaseModel):
    width: int = Field(default=1280, ge=320, le=3840)
    height: int = Field(default=720, ge=240, le=2160)

class CaptureConfig(BaseModel):
    screenshot: bool = False
    screenshot_full_page: bool = True
    pdf: bool = False
    pdf_format: Literal["A4", "Letter", "Legal"] = "A4"
```

### 2.3 Design Rationale

**Extend web_scrape rather than new plugin**: Both modes share URL validation, SSRF prevention, AuditedHTTPClient, fingerprinting, and the same output contract (`content_field`, `fingerprint_field`, `fetch_status`, `fetch_url_final`).

**Declarative interactions, not arbitrary JS**: `InteractionStep` is intentionally restrictive (click, scroll, fill, wait, screenshot). No `evaluate()` for arbitrary JavaScript. Each step is auditable and config-time validatable.

**Optional dependency**: Playwright (~150MB Chromium) is in an optional group `browser = ["playwright>=1.40"]`. Config validation at `on_start()` crashes with a clear error if `mode="browser"` but Playwright isn't installed.

### 2.4 Security Model

| Risk | Mitigation |
|------|------------|
| JavaScript execution | Sandboxed browser process. `block_resources` default blocks external scripts |
| SSRF via redirects | Playwright `route()` API intercepts all navigation — same IP pinning as HTTP mode |
| Cookie/credential leakage | Fresh browser context per URL (no persistent state) |
| Resource exhaustion | `wait_timeout_ms` caps load time. Browser pool limits concurrent instances |
| Exfiltration via JS | Network interception blocks outbound requests to non-target domains |

### 2.5 Execution Model

Browser mode uses `BatchTransformMixin` (streaming) for concurrent browser contexts, same as HTTP mode already does. The browser pool acts as the concurrency limiter (replaces the HTTP connection pool):

- `pool_size` config controls max concurrent browser contexts (default: 4, lower than HTTP default due to heavier resource cost)
- Each row acquires a context, processes, and releases — FIFO output ordering via `RowReorderBuffer`
- Backpressure: orchestrator blocks when pool is full

### 2.6 Browser Lifecycle

```
on_start() → Launch Playwright browser → Create browser pool (max pool_size contexts)
per row   → Acquire context → Validate URL → Navigate with route interception
          → Wait → Execute interactions → Extract content → Fingerprint
          → Capture screenshot/PDF (if configured) → Record to audit trail → Close context
close()   → Drain pool → Shut down browser
```

### 2.6 Visual Capture

Screenshot and PDF capture stored as PayloadStore blob references (not inline in pipeline rows):
- Output fields: `{prefix}_screenshot_ref`, `{prefix}_pdf_ref`
- Same artifact pattern as existing web_scrape response body storage

---

## 3. Report Sink

### 3.1 Configuration

```python
class ReportSinkConfig(SinkPathConfig):
    format: Literal["markdown", "html", "pdf", "json_report"] = "markdown"
    template: str | None = None                # Jinja2 template path or inline
    title: str = "Pipeline Report"

    sections: Sequence[ReportSection] | None = None
    include_summary: bool = True
    include_field_stats: bool = False
    max_rows_in_report: int | None = None
    sort_by: str | None = None
    sort_order: Literal["asc", "desc"] = "asc"

class ReportSection(BaseModel):
    heading: str
    fields: Sequence[str]
    format: Literal["table", "list", "cards"] = "table"
    group_by: str | None = None
```

### 3.2 Output Formats

| Format | Output | Implementation |
|--------|--------|----------------|
| `markdown` | `.md` file | Jinja2 → Markdown |
| `html` | `.html` file | Jinja2 → HTML with embedded CSS |
| `pdf` | `.pdf` file | Jinja2 → HTML → WeasyPrint |
| `json_report` | `.json` file | Structured JSON for downstream consumption |

### 3.3 Template Context

When using Jinja2 templates, the template receives:

```python
{
    "title": str,
    "rows": list[dict],           # All pipeline rows
    "summary": {                  # If include_summary=True
        "total_rows": int,
        "timestamp": str,
        "duration_seconds": float,
        "source_plugin": str,
    },
    "field_stats": dict,          # If include_field_stats=True
    "metadata": dict,             # Pipeline metadata
}
```

### 3.4 PDF Dependency

WeasyPrint for PDF generation, in optional dependency group:

```toml
[project.optional-dependencies]
reports = ["weasyprint>=60", "jinja2>=3.1"]
```

---

## 4. Notification Sink

### 4.1 Configuration

```python
class NotificationSinkConfig(PluginConfig):
    channel: Literal["email", "slack", "teams", "webhook"]
    channel_config: Mapping[str, Any]          # Channel-specific settings
    template: str | None = None                # Jinja2 message template
    on_condition: Literal["always", "non_empty", "threshold"] = "always"
    threshold_field: str | None = None         # For on_condition="threshold"
    threshold_value: float | None = None
```

### 4.2 Channels

| Channel | Config | Transport |
|---------|--------|-----------|
| `email` | smtp_host, smtp_port, from_addr, to_addrs, subject_template | SMTP via `smtplib` |
| `slack` | webhook_url | HTTP POST (AuditedHTTPClient) |
| `teams` | webhook_url | HTTP POST (AuditedHTTPClient) |
| `webhook` | url, method, headers | HTTP (AuditedHTTPClient) |

All HTTP-based channels use `AuditedHTTPClient` for audit trail recording.

---

## 5. Phased Roadmap

### Phase 1: Web Research Pipeline (Core)

| Component | Type | Effort |
|-----------|------|--------|
| Search protocol + types | Infrastructure (contracts layer) | Medium |
| OpenSearch provider | Provider (source + sink + RAG) | Large |
| Enhanced web_scrape (browser mode) | Transform enhancement | Large |
| Report sink (Markdown, HTML, PDF, JSON) | New sink | Medium |
| Chroma full provider upgrade | Provider enhancement | Small |

**Validation**: Scrape competitor sites (browser mode) → LLM extract → index OpenSearch → query for trends → generate HTML/PDF report.

### Phase 2: Vector Search + Notifications

| Component | Type | Effort |
|-----------|------|--------|
| Qdrant provider | Provider (source + sink + RAG) | Medium |
| Notification sink | New sink | Medium |
| RSS/Atom source | New source | Small |

**Validation**: RSS feed → filter → RAG against Qdrant → summarize → Slack notification.

### Phase 3: Document Processing + Lightweight Search

| Component | Type | Effort |
|-----------|------|--------|
| Meilisearch provider | Provider (source + sink + RAG) | Medium |
| Document source (PDF, DOCX, PPTX) | New source | Medium |
| Chunking transform | New transform | Medium |
| Embedding transform | New transform (provider registry) | Medium |

**Validation**: Ingest PDFs → chunk → embed → index Meilisearch/Qdrant → RAG Q&A.

### Phase 4: Advanced Analytics + Automation

| Component | Type | Effort |
|-----------|------|--------|
| Scheduled pipeline triggers | Infrastructure | Medium |
| Diff/change detection transform | New transform | Medium |
| Entity extraction transform | New transform | Medium |
| Dashboard sink | New sink | Medium |

**Validation**: Scheduled competitor scrape → diff detection → entity extraction → dashboard + alerts.

### Dependency Graph

```
Phase 1 ──── Search protocol + OpenSearch + web_scrape browser + report sink + Chroma upgrade
  ↓
Phase 2 ──── Qdrant + notification sink + RSS source
  ↓
Phase 3 ──── Meilisearch + document source + chunking + embedding
  ↓
Phase 4 ──── Scheduled triggers + diff detection + entity extraction + dashboard sink
```

Each phase is independently shippable. Phase 1 validates the architecture. Subsequent phases build on it without modifying the core abstractions.

---

## 6. Layer Placement

Per ELSPETH's 4-layer model:

| Component | Layer | Rationale |
|-----------|-------|-----------|
| Search protocol, query types, capability types | L0 (contracts) | Shared types, no upward imports |
| Search provider interface | L0 (contracts) | Protocol definition |
| OpenSearch/Qdrant/Meilisearch providers | L3 (plugins) | External service integration |
| search_source, search_sink | L3 (plugins) | Plugin implementations |
| Search RAG provider | L3 (plugins) | Added to RAG transform's provider registry |
| Browser mode (web_scrape) | L3 (plugins) | Extends existing transform |
| Report sink, notification sink | L3 (plugins) | New sink plugins |
| Chunking, embedding, entity extraction | L3 (plugins) | New transforms |

No new layer violations. All new code sits in L0 (contracts) or L3 (plugins).

---

## 7. Verification Plan

### Phase 1 Verification

1. **Unit tests**: Each search data type (frozen, deep-freeze), query builder, filter tree serialization, capability validation
2. **Integration tests**: OpenSearch provider against containerized OpenSearch (Docker Compose). Index → search → scroll → aggregation round-trip
3. **End-to-end test**: Full web research pipeline (scrape → LLM → index → query → report) using `ExecutionGraph.from_plugin_instances()` per CLAUDE.md test path rules
4. **Browser tests**: Playwright mode against a local test server with JS-rendered content
5. **Report tests**: Verify Markdown/HTML/PDF output against known input data
6. **Tier model enforcement**: Run `enforce_tier_model.py` — all new code must pass
7. **Config contracts**: Run `check_contracts` — new Settings fields must have contract coverage

### Composer Integration

New plugins are automatically discoverable by the composer (pluggy auto-registration). Verify:
- `list_sources` returns `search_source` (once registered)
- `list_sinks` returns `search_sink`, `report`, `notification`
- `get_plugin_schema("source", "search")` returns full JSON schema
- Pipeline composer can build a web research pipeline conversationally
