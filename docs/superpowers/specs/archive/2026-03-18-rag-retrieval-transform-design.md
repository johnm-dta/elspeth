# RAG Retrieval Transform Plugin — Design Spec

**Date:** 2026-03-18
**Status:** Reviewed (R7 — R6 issues resolved; R7 fixes: ctx.fingerprint_key hallucination, on_no_results warning, PluginRetryableError enforcement, on_start failure spec, ImmutableSandboxedEnvironment shared infra, index name validation, 401 retry semantics, search_mode/min_score warning, state_id guard pattern, Settings→Runtime note)
**Scope:** New `rag_retrieval` transform plugin, `RetrievalProvider` protocol, Azure AI Search provider implementation

## Overview

Add a Retrieval-Augmented Generation (RAG) retrieval transform to ELSPETH. This plugin retrieves relevant context from a vector/search backend and attaches it to the pipeline row as prefixed fields. A downstream LLM transform then references the retrieved context in its prompt template.

This is a **retrieval-only** transform — it does not perform generation. The separation follows ELSPETH's composability model: retrieval and generation are distinct operations with different error modes, retry semantics, and audit requirements. They compose naturally as separate DAG steps.

## Motivation

LLM transforms in ELSPETH currently operate on row data alone. For many use cases — classification, summarization, Q&A — the LLM needs domain-specific context that isn't in the row. RAG provides this by retrieving relevant documents/chunks at query time and injecting them into the prompt.

Building RAG as a separate retrieval transform (rather than bolting it onto the LLM transform) provides:
- **Independent auditability** — "what context was retrieved" and "what the LLM did with it" are separate Landscape entries
- **Composability** — gates between retrieval and generation (e.g., filter low-relevance results), multiple retrieval steps with different knowledge bases, retrieval without generation (similarity search pipelines)
- **Reuse** — the retrieval transform is useful beyond LLM augmentation (deduplication, nearest-neighbor lookup, document matching)

## Architecture

### File Layout

```
src/elspeth/plugins/
├── infrastructure/
│   └── clients/
│       └── retrieval/
│           ├── __init__.py        # Public exports
│           ├── base.py            # RetrievalProvider protocol
│           ├── azure_search.py    # Azure AI Search implementation
│           └── types.py           # RetrievalChunk, RetrievalResult dataclasses
└── transforms/
    └── rag/
        ├── __init__.py            # Plugin registration
        ├── transform.py           # RAGRetrievalTransform
        ├── config.py              # RAGRetrievalConfig (Pydantic)
        ├── query.py               # Query construction strategies
        └── formatter.py           # Context output formatting
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `RetrievalProvider` | Protocol for search backends — search interface + lifecycle |
| `AzureSearchProvider` | Azure AI Search implementation of `RetrievalProvider` |
| `RAGRetrievalTransform` | Pipeline integration — query construction, retrieval orchestration, output attachment |
| `query.py` | Query construction from row data (field, template, regex modes) |
| `formatter.py` | Context formatting (numbered, separated, raw) with length capping |

### Layer Placement

- `RetrievalProvider` protocol and types: `plugins/infrastructure/clients/retrieval/` (L3)
- `AzureSearchProvider`: `plugins/infrastructure/clients/retrieval/` (L3)
- `RAGRetrievalTransform`: `plugins/transforms/rag/` (L3)

No cross-layer violations. The retrieval provider lives alongside `AuditedHTTPClient` and `AuditedLLMClient` in the client infrastructure.

**Placement rationale:** The `RetrievalProvider` protocol and implementations are placed in `plugins/infrastructure/clients/retrieval/` rather than under `plugins/transforms/rag_retrieval/` (the `LLMProvider` precedent). This placement reflects the intent for retrieval providers to be reusable across multiple transform types — unlike LLM providers which are tightly coupled to the LLM transform's query/response contract. Both locations are L3 and architecturally equivalent.

### Relationship to Existing LLM Transform

The RAG retrieval transform and LLM transform are **independent plugins** with no code-level coupling. They compose via the pipeline DAG:

```
Source → RAGRetrievalTransform → LLMTransform → Sink
                |                       |
         attaches context        references context
         as row fields           via {{ row.prefix__rag_context }}
```

The LLM transform doesn't know or care where `prefix__rag_context` came from — it's just a row field available in the template. This is the same composition model as any other transform chain.

## RetrievalProvider Protocol

```python
@runtime_checkable
class RetrievalProvider(Protocol):
    """Search backend interface for RAG retrieval."""

    def search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        *,
        state_id: str,
        token_id: str | None,
    ) -> list[RetrievalChunk]:
        """Execute a search query and return ranked results.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold (0.0-1.0).
                Results below this score are discarded.
            state_id: Per-row audit identity — passed through to AuditedHTTPClient
                for correctly-scoped call recording. This follows the LLMProvider
                pattern where execute_query() takes state_id and token_id at the
                call site.
            token_id: Pipeline token identity for audit correlation.

        Returns:
            List of RetrievalChunk, ordered by descending relevance score.
            May be empty if no results meet the min_score threshold.
        """
        ...

    def close(self) -> None:
        """Release provider resources (connections, clients)."""
        ...
```

### RetrievalChunk

```python
@dataclass(frozen=True)
class RetrievalChunk:
    """A single retrieved document chunk."""

    content: str           # The retrieved text content
    score: float           # Relevance score, normalized to 0.0-1.0
    source_id: str         # Document/chunk identifier (for audit traceability)
    metadata: dict[str, Any]  # Provider-specific metadata (page, section, index name, etc.)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"Score must be normalized to [0.0, 1.0], got {self.score!r}. "
                f"Provider score normalization bug — check the provider implementation."
            )
        # Validate metadata is JSON-serializable — catches provider bugs at construction
        # time rather than at row assembly time when __rag_sources is serialized.
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"metadata must be JSON-serializable (got {type(exc).__name__}: {exc}). "
                f"Provider must coerce non-primitive types (datetime → ISO 8601 str, "
                f"UUID → str, etc.) at the Tier 3 boundary before constructing RetrievalChunk."
            ) from exc
```

The `__post_init__` assertions enforce two contracts at construction time, consistent with ELSPETH's offensive programming stance:
1. **Score normalization:** A provider that returns an out-of-range score crashes immediately with a diagnostic message rather than silently passing bad data downstream.
2. **Metadata serializability:** The `metadata` dict is opaque but will be JSON-serialized into `{prefix}__rag_sources`. Non-serializable types (datetime, UUID, bytes) from external API responses must be coerced to JSON primitives at the Tier 3 boundary in the provider — this is legal per CLAUDE.md's coercion rules (external data at trust boundary). The `json.dumps()` probe catches provider bugs at `RetrievalChunk` construction, not two call frames away during row assembly.

### Design Rationale

The protocol is deliberately minimal:
- **`search()` takes primitives plus audit identity** — no provider-specific query objects leak into the transform. The `state_id` and `token_id` keyword arguments follow the `LLMProvider.execute_query()` pattern, enabling correctly-scoped audit recording per row.
- **Scores are normalized to 0.0-1.0** — each provider handles its own score normalization internally (Azure AI Search returns different scales for different search modes). **Score comparability note:** `min_score` thresholds are not portable across `search_mode` changes — different modes produce different score distributions even after normalization. Operators must recalibrate `min_score` when changing `search_mode`. The `search_mode` is recorded in `success_reason_json` for per-run audit traceability.
- **`metadata` is opaque but JSON-serializable** — provider-specific details (page numbers, section headers, chunk IDs) travel through without the protocol needing to enumerate them. The JSON-serializability constraint is enforced at `RetrievalChunk` construction time.
- **No embedding method on the protocol** — the provider handles embedding internally. Some backends (Azure AI Search with integrated vectorization) handle embedding server-side; others need client-side embedding. This is a provider concern, not a protocol concern.

## Azure AI Search Provider

### Day-One Implementation

```python
class AzureSearchProvider:
    """Azure AI Search implementation of RetrievalProvider."""

    def __init__(
        self,
        config: AzureSearchProviderConfig,
        *,
        recorder: LandscapeRecorder,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        limiter: RateLimiter | NoOpLimiter | None = None,
    ) -> None:
        ...

    def search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        *,
        state_id: str,
        token_id: str | None,
    ) -> list[RetrievalChunk]:
        # Construct per-call AuditedHTTPClient scoped to this row's state_id.
        # NOTE: Initialize http_client before try block to avoid UnboundLocalError
        # in finally if AuditedHTTPClient.__init__ raises (e.g., FrameworkBugError
        # from missing fingerprint key).
        http_client = None
        http_client = AuditedHTTPClient(
            recorder=self._recorder,
            state_id=state_id,
            run_id=self._run_id,
            telemetry_emit=self._telemetry_emit,
            limiter=self._limiter,
            token_id=token_id,
            base_url=self._config.endpoint,
        )
        try:
            # Execute search via HTTP client (audit recorded automatically)
            ...
            # Wrap RetrievalChunk construction — __post_init__ raises ValueError
            # for out-of-range scores or non-serializable metadata. Translate to
            # RetrievalError at the Tier 3 boundary to avoid crashing the run.
            try:
                chunks = [RetrievalChunk(...) for item in results]
            except ValueError as exc:
                raise RetrievalError(
                    f"Provider returned invalid data: {exc}",
                    retryable=False,
                ) from exc
        finally:
            if http_client is not None:
                http_client.close()
```

The provider constructs a **per-call `AuditedHTTPClient`** scoped to the current row's `state_id`, following the `WebScrapeTransform` pattern at `web_scrape.py:383`. This ensures every search call is recorded under the correct row's audit identity. The provider holds shared resources (`recorder`, `run_id`, `limiter`) at construction time but creates correctly-scoped HTTP clients per `search()` invocation.

**Why per-call construction:** `AuditedHTTPClient` binds `state_id` at construction with no per-call override (unlike `token_id` which has `token_id_override`). A single shared instance would record all calls under the first row's `state_id` — silent audit corruption. The per-call pattern avoids this entirely. Connection overhead is minimal: `httpx.Client` uses connection pooling internally, and search calls are I/O-bound, not connection-setup-bound.

**Known Limitations:** Each row creates a new `AuditedHTTPClient` with a fresh `httpx.Client` and connection pool. No TCP connections are reused across rows. At high throughput (>500 rows/minute), TLS handshake overhead adds measurable latency. This follows the established `WebScrapeTransform` pattern. If profiling shows retrieval latency is the bottleneck, the transform can be upgraded to hold a persistent `httpx.Client` on the provider instance with per-request `state_id` override support.

### Features

- **Search modes:** `vector`, `keyword`, `hybrid` (vector + keyword), `semantic` (semantic ranking)
- **Integrated vectorization:** When `search_mode` is `vector` or `hybrid`, the provider uses Azure AI Search's built-in vectorizer (configured on the search index). No client-side embedding needed.
- **Score normalization:** Azure returns different score ranges per search mode. The provider normalizes all scores to 0.0-1.0.
- **Auth:** API key or managed identity (via `azure-identity`)
- **Audit trail:** All search API calls go through `AuditedHTTPClient` — automatically recorded in Landscape

### Configuration

```python
class AzureSearchProviderConfig(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    endpoint: str              # https://<service>.search.windows.net
    index: str                 # Search index name (validated: alphanumeric, hyphens, underscores only)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """HTTPS required; reject private/reserved IP ranges to prevent SSRF."""
        from elspeth.core.security.web import validate_url_for_ssrf
        parsed = urllib.parse.urlparse(v)
        if parsed.scheme != "https":
            raise ValueError(f"endpoint must use HTTPS scheme, got {parsed.scheme!r}")
        validate_url_for_ssrf(v)
        return v
        # Although endpoints are operator-authored config (Tier 2), ELSPETH's security
        # posture requires URL validation at all external call boundaries to prevent
        # SSRF via misconfiguration.

    @field_validator("index")
    @classmethod
    def validate_index_name(cls, v: str) -> str:
        """Index name goes into URL path /indexes/{index}/docs/search — validate for path safety."""
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$', v):
            raise ValueError(
                f"index must contain only alphanumeric characters, hyphens, and underscores "
                f"(and start with alphanumeric), got {v!r}. The index name is interpolated "
                f"into a URL path — special characters could produce malformed requests."
            )
        return v

    api_key: str | None = None # API key (alternative to managed identity)
    use_managed_identity: bool = False
    api_version: str = "2024-07-01"

    search_mode: Literal["vector", "keyword", "hybrid", "semantic"] = "hybrid"
    # WARNING: min_score thresholds are NOT portable across search_mode changes.
    # Different modes produce fundamentally different score distributions even after
    # normalization. A min_score of 0.7 calibrated for "hybrid" will filter differently
    # under "vector" or "keyword". Recalibrate min_score whenever you change search_mode.
    request_timeout: float = 30.0        # HTTP request timeout in seconds (semantic ranking can exceed default)

    # Vector search options (when search_mode is vector or hybrid)
    vector_field: str = "contentVector"   # Name of the vector field in the index
    semantic_config: str | None = None    # Semantic configuration name (for semantic mode)

    @model_validator(mode="after")
    def validate_auth(self) -> Self:
        if not self.api_key and not self.use_managed_identity:
            raise ValueError("Specify either api_key or use_managed_identity=true")
        if self.api_key and self.use_managed_identity:
            raise ValueError("Specify only one of api_key or use_managed_identity")
        return self

    @model_validator(mode="after")
    def validate_semantic_config(self) -> Self:
        if self.search_mode == "semantic" and not self.semantic_config:
            raise ValueError("semantic search_mode requires semantic_config")
        return self
```

### Tier 3 Boundary

Azure AI Search responses are Tier 3 (external data). Validation at the boundary:

1. **JSON parse** — reject malformed responses
2. **Response structure** — `value` array present, each item has `@search.score` and expected content fields
3. **Score normalization** — map provider-specific scores to 0.0-1.0 range
4. **Content extraction** — extract text content from configured field, reject items with missing content
5. **Metadata coercion** — coerce non-JSON-primitive types to strings before constructing `RetrievalChunk`. Azure responses may include datetime values, UUIDs, or nested objects in metadata fields. Coercion at the Tier 3 boundary is permitted per CLAUDE.md (external data at trust boundary): `datetime` → ISO 8601 string, `UUID` → string, other non-primitive types → `str()`. This ensures `RetrievalChunk.metadata` passes the JSON-serializability check in `__post_init__`.

**CRITICAL: `RetrievalChunk.__post_init__` ValueError handling.** The `__post_init__` validation (score range, metadata serializability) is correct offensive programming, but the `ValueError` it raises is not a `PluginRetryableError` and will escape all processor catch clauses, crashing the run instead of quarantining the row. `AzureSearchProvider.search()` MUST wrap `RetrievalChunk(...)` construction in a try/except for `ValueError` and convert it to `RetrievalError(retryable=False)` with a descriptive message. The fix is to translate the `ValueError` into a `RetrievalError` at the Tier 3 boundary inside `search()`.

Once validated, `RetrievalChunk` objects are Tier 2 — trusted by the transform.

## RAGRetrievalTransform

### Extends BaseTransform (Synchronous)

```python
class RAGRetrievalTransform(BaseTransform):
    name = "rag_retrieval"
    determinism = Determinism.EXTERNAL_CALL
```

This transform uses the synchronous `process()` model, NOT `BatchTransformMixin`. `BatchTransformMixin` provides *within-row* concurrency (multiple sub-tasks per row, e.g., multi-query LLM calls), not across-row concurrency. A single retrieval query per row gains nothing from the mixin — it would add OutputPort wiring, reorder buffer, and worker pool overhead for zero benefit.

If a future use case requires multiple concurrent retrievals per row (e.g., querying multiple indexes simultaneously for a single row), the transform can be upgraded to `BatchTransformMixin` at that point. For the day-one single-query-per-row design, synchronous `process()` is correct and simpler.

### Configuration

```python
# Unified provider registry — maps provider names to (config_class, provider_class) tuples.
# Single source of truth for both config parsing AND provider instantiation.
# Follows the LLM transform's _PROVIDERS pattern to eliminate the sync failure mode
# of maintaining two separate dispatch tables.
_PROVIDERS: dict[str, tuple[type[BaseModel], type]] = {
    "azure_search": (AzureSearchProviderConfig, AzureSearchProvider),
}

class RAGRetrievalConfig(TransformDataConfig):
    # Output field prefix (mandatory)
    output_prefix: str                # e.g., "financial" → "financial__rag_context"

    # Query construction (query_field is always required)
    query_field: str                  # Row field to use as search query source
    query_template: str | None = None # Jinja2 template ({{ query }} = extracted field value)
    query_pattern: str | None = None  # Regex to extract search text from field value

    # Provider selection and config
    provider: str                     # Provider name (e.g., "azure_search")
    provider_config: dict[str, Any]   # Validated eagerly against provider's config class

    # Retrieval parameters
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Filtering semantics: results with score >= min_score are kept; results with
    # score < min_score are discarded. The comparison is inclusive — a chunk with
    # exactly score=1.0 when min_score=1.0 is kept.

    # Zero results behavior
    on_no_results: Literal["quarantine", "continue"] = "quarantine"
    # WARNING: "continue" mode enables silent semantic degradation. Rows with zero retrieval
    # results proceed to downstream transforms (e.g., LLM) with empty context fields. The LLM
    # will produce plausible-looking output with no grounding — completing as COMPLETED in the
    # Landscape. The only way to detect this is querying success_reason_json for
    # retrieval_status="empty". STRONGLY RECOMMENDED: pair on_no_results="continue" with a
    # downstream gate that routes on {prefix}__rag_count == 0 to prevent contextless LLM output.

    # Context formatting
    context_format: Literal["numbered", "separated", "raw"] = "numbered"
    context_separator: str = "\n---\n"  # Used when context_format="separated"
    max_context_length: int | None = Field(default=None, ge=1)
    # Character cap. Must be at least 1 if specified. Use None (the default) for no truncation limit.

    @field_validator("output_prefix")
    @classmethod
    def validate_prefix(cls, v: str) -> str:
        import keyword
        if not v.isidentifier():
            raise ValueError(f"output_prefix must be a valid Python identifier, got {v!r}")
        if keyword.iskeyword(v):
            raise ValueError(
                f"output_prefix must not be a Python keyword, got {v!r}. "
                f"Keywords like 'class', 'return' produce field names that break Jinja2 templates."
            )
        return v

    @model_validator(mode="after")
    def validate_query_modes(self) -> Self:
        if self.query_template and self.query_pattern:
            raise ValueError("query_template and query_pattern are mutually exclusive")
        return self

    @model_validator(mode="after")
    def validate_provider_config(self) -> Self:
        provider_entry = _PROVIDERS.get(self.provider)
        if provider_entry is None:
            raise ValueError(f"Unknown provider: {self.provider!r}. Available: {sorted(_PROVIDERS)}")
        config_cls, _ = provider_entry
        config_cls(**self.provider_config)  # Eagerly validate provider config at YAML load time
        return self

    @field_validator("query_pattern")
    @classmethod
    def validate_regex(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}") from e
        return v
```

Note: `TransformDataConfig` extends `DataPluginConfig`, which requires `schema_config`. `TransformDataConfig` also provides `required_input_fields` — the config MUST include `query_field` in `required_input_fields` so the DAG builder can statically verify that the upstream source or transform guarantees this field. If `query_field` is somehow absent at runtime despite DAG validation (which would indicate a DAG validator bug or source schema change), `row[query_field]` raises `KeyError` — this is correct crash behavior per the tier model (Tier 2 data, missing field = upstream contract violation, not a row to quarantine).

Rate limiting uses the settings-level `RateLimitRegistry`, not a per-plugin config field. The unified provider registry eagerly validates `provider_config` against the correct Pydantic model at YAML load time (fail-fast, not deferred to first row). Provider instantiation in `on_start()` also uses this registry to look up the provider class — no separate dispatch table.

**Note on Settings→Runtime pattern:** CLAUDE.md's Settings→Runtime configuration pattern (Pydantic Settings → frozen `Runtime*Config` dataclass via `from_settings()`) applies to **engine-level** configuration (e.g., `RetrySettings` → `RuntimeRetryConfig`). Plugin-level configs like `RAGRetrievalConfig` are Pydantic models used directly at runtime — this is consistent with how `LLMTransformConfig` and other plugin configs work. No `RuntimeRAGConfig` dataclass is needed. The `self._validated_provider_config` stored in `on_start()` is a validated Pydantic model, not a Settings→Runtime conversion.

**Double validation note:** The `provider_config` dict is validated twice: once at YAML load time (in `validate_provider_config` model validator) and once at `on_start()` (when constructing the provider). To avoid the `.get()` anti-pattern on validated data and to eliminate the redundant second parse, `on_start()` stores the validated Pydantic model as `self._validated_provider_config`. This attribute is used in `process()` for typed attribute access (e.g., `self._validated_provider_config.search_mode`) instead of dict key access with defaults — consistent with CLAUDE.md's offensive programming stance.

### Query Construction

Three modes, all anchored on `query_field`:

**1. Field only** (`query_field` set, no template/regex):
```python
query = row[query_field]  # Use field value verbatim
```

**2. Field + template** (`query_field` + `query_template`):
```python
extracted = row[query_field]
query = self._compiled_template.render(query=extracted, row=row.to_dict())
# Template has access to {{ query }} (the extracted value) and {{ row }} (full row)
# Uses ImmutableSandboxedEnvironment from plugins/infrastructure/templates.py (shared infra)
```

Template is pre-compiled at `__init__` time via `ImmutableSandboxedEnvironment.from_string(query_template)` with `StrictUndefined` (structural errors fail the run at setup). Render errors at row time produce `TransformResult.error()` (quarantine the row). We use `ImmutableSandboxedEnvironment` from `plugins/infrastructure/templates.py` (shared infra) rather than `PromptTemplate` because `PromptTemplate.render()` accepts only `(row, *, contract)` — it does not support injecting a separate `{{ query }}` context variable.

**3. Field + regex** (`query_field` + `query_pattern`):
```python
extracted = row[query_field]
# Note: extracted is already validated non-None by step 2
try:
    # Python re.search() has no timeout parameter — wrap in ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(compiled_pattern.search, str(extracted))
        match = future.result(timeout=5.0)
except concurrent.futures.TimeoutError:
    return TransformResult.error(
        {"reason": "no_regex_match", "field": query_field, "cause": "regex_timeout"},
        retryable=False,
    )
if match is None:
    return TransformResult.error({"reason": "no_regex_match", ...}, retryable=False)
captured = match.group(1) if match.lastindex else match.group(0)
if captured is None:
    # Capture group defined but didn't participate (e.g., optional group in alternation)
    return TransformResult.error(
        {"reason": "no_regex_match", "field": query_field, "cause": "capture_group_empty"},
        retryable=False,
    )
query = captured
```

Regex is pre-compiled at `__init__` time. No match → row-level error (quarantine). If the regex has capture groups, the first group is used; otherwise the full match. **Non-participating capture groups** (where `match.lastindex` is set but `match.group(1)` returns `None`, e.g., optional groups like `(?:issue|problem)(?::\s*(.+?))?`) are treated as no-match and quarantined with cause `"capture_group_empty"`. This prevents `None` from being silently passed as a query string to the provider.

**ReDoS protection:** The regex search uses a **5-second timeout** to prevent catastrophic backtracking on adversarial or pathological input. Python 3.11+ does not natively support `re.search()` timeout, so the implementation wraps the regex execution in a thread with `concurrent.futures.ThreadPoolExecutor` and a 5-second deadline. If the regex exceeds the timeout, the row is quarantined with `"cause": "regex_timeout"`. Pipeline architects should validate that their patterns are not vulnerable to backtracking (e.g., avoid `(a+)+` patterns) — the timeout is a safety net, not a substitute for safe patterns.

**Thread pool caching (MUST):** The `ThreadPoolExecutor(max_workers=1)` MUST be cached at the transform level (created in `__init__`, closed in `close()`) — NOT created per-row. At high throughput (hundreds of rows per minute), per-row executor creation causes unnecessary thread churn and risks OS thread exhaustion (`RuntimeError: can't start new thread`). This is a blocking implementation requirement, not a future optimization. The `close()` method must call `self._regex_executor.shutdown(wait=False)` if the executor was created (guard on `self._regex_executor is not None` for pipelines that don't use regex query mode).

### Plugin Lifecycle

```
__init__(config)
  → Validate config (Pydantic, including eager provider_config validation)
  → Pre-compile query_template (if provided) — uses ImmutableSandboxedEnvironment
    with StrictUndefined from plugins/infrastructure/templates.py (shared infra,
    NOT from plugins/transforms/llm/templates.py). NOT PromptTemplate, whose
    render() signature accepts only row + contract. The template is compiled once
    via env.from_string(query_template). TemplateSyntaxError fails the run at setup.
  → Pre-compile query_pattern (if provided) — already validated by Pydantic
  → Compute declared_output_fields from output_prefix
  → Compute _output_schema_config with guaranteed_fields listing all four output fields
    ({prefix}__rag_context, {prefix}__rag_score, {prefix}__rag_count, {prefix}__rag_sources).
    This enables DAG contract validation — downstream transforms that declare
    required_input_fields can statically verify their dependencies at DAG construction time.
  → NOTE: Provider is NOT constructed here (needs landscape/run_id from context)

on_start(ctx: LifecycleContext)
  → Call super().on_start(ctx) FIRST — sets the BaseTransform lifecycle flag.
    TransformExecutor checks this flag; omitting super() causes a lifecycle check failure.
  → Store ctx.landscape (LandscapeRecorder), ctx.run_id, ctx.telemetry_emit, ctx.rate_limit_registry
  → Look up (config_cls, provider_cls) from unified _PROVIDERS registry
  → Construct RetrievalProvider (provider_cls) with validated provider_config,
    recorder=ctx.landscape, run_id=ctx.run_id, telemetry_emit=ctx.telemetry_emit,
    limiter=ctx.rate_limit_registry.get_limiter("azure_search") if ctx.rate_limit_registry is not None else None
  → Provider is constructed with shared resources but does NOT hold an AuditedHTTPClient.
    Per-call AuditedHTTPClient is constructed inside search() using the state_id passed
    at call time (see AzureSearchProvider section).
  → Provider performs connection validation (e.g., index exists check).
    On failure: raise RetrievalError(retryable=False, status_code=...) — this crashes the run
    at setup time, which is correct. A missing or misconfigured index can never produce valid
    results for any row. The orchestrator records the exception in the run's lifecycle events.
    The error message MUST include the index name and endpoint for operator diagnosis.

process(row, ctx: TransformContext)
  → See Process Flow below
  → Calls provider.search(..., state_id=ctx.state_id,
      token_id=ctx.token.token_id if ctx.token is not None else None)
  → Each search call constructs a correctly-scoped AuditedHTTPClient inside the provider

on_complete(ctx: LifecycleContext)
  → Emit retrieval statistics via ctx.telemetry_emit (total queries, chunks retrieved,
    mean score, quarantine count). Do NOT use logger — per CLAUDE.md logging policy,
    operational statistics go through telemetry, not logging.
  → Guard mean score calculation: if total_queries == 0, emit None for mean_score
    (avoids StatisticsError on empty sequence).

close()
  → Guard: if self._provider is not None, call provider.close() to release shared resources.
    The orchestrator may call close() even if on_start() never ran (e.g., config validation
    failure during DAG construction). Initialise self._provider = None in __init__ so close()
    is safe regardless of lifecycle ordering. This is a legitimate lifecycle guard, not
    defensive programming — the engine does not guarantee on_start() precedes close().
```

**Template sandbox:** Query templates use `ImmutableSandboxedEnvironment` with `StrictUndefined`. **Prerequisite: move `ImmutableSandboxedEnvironment` from `plugins/transforms/llm/templates.py` to `plugins/infrastructure/templates.py` as shared infrastructure.** The RAG transform importing from the LLM plugin's private module creates cross-plugin coupling — if the LLM template module is restructured, the RAG transform silently breaks. Moving it to `plugins/infrastructure/` reflects its actual role: shared template infrastructure used by multiple transform types. Both the LLM transform and RAG transform then import from the same shared location.

We do NOT use `PromptTemplate.render()` because its signature is `render(row, *, contract=None)` — it does not support injecting a separate `{{ query }}` context variable. Instead, the transform compiles the template at `__init__` time via `env.from_string(query_template)` and renders at row time via `template.render(query=extracted, row=row.to_dict())`. This provides the same sandbox security (no attribute access, no method calls, no module imports) while supporting the `{{ query }}` + `{{ row }}` context variables the spec requires.

**Note on `ImmutableSandboxedEnvironment` limitations:** The sandbox blocks imports, attribute access, and method calls but does NOT limit CPU or memory consumption from template loops (e.g., `{% for i in range(10**9) %}`). Since query templates are written by the pipeline architect (trusted config), this is acceptable. If user-generated templates are ever supported, a template execution timeout would be needed.

**`_output_schema_config` and DAG validation:** The transform exposes its four output fields as `guaranteed_fields` via `_output_schema_config`, following the LLM transform pattern. This enables the DAG builder to statically verify that downstream transforms' `required_input_fields` (including LLM template variable dependencies like `{{ row.policy__rag_context }}`) are satisfied at DAG construction time — not deferred to runtime `UndefinedError`.

### Process Flow (per row)

```
0. Guard: if ctx.state_id is None: raise RuntimeError("ctx.state_id not set by executor")
   (matches WebScrapeTransform pattern at web_scrape.py:378 — uses explicit if/raise, NOT assert,
   because assertions are stripped with -O)
1. Extract query field value from row
2. Validate extracted value is not None
   → If None: TransformResult.error({"reason": "invalid_input", "field": query_field,
     "cause": "null_value"}, retryable=False) (quarantine)
   This prevents silent coercion of None to the literal string "None" in the regex
   path (via str(extracted)) or passing None to the template/provider.
3. Construct search query (field / template / regex)
   → On regex no-match or timeout: TransformResult.error(retryable=False) (quarantine)
   → On template render error: TransformResult.error(retryable=False) (quarantine)
4. Validate constructed query is non-empty and non-whitespace
   → If empty/whitespace: TransformResult.error({"reason": "invalid_input",
     "field": query_field, "cause": "empty_query"}, retryable=False) (quarantine)
   This prevents sending empty queries to the provider, which may return HTTP 400
   (misleading "retrieval_failed") or a full result set (silent semantic error).
5. Call provider.search(query, top_k, min_score,
     state_id=ctx.state_id,
     token_id=ctx.token.token_id if ctx.token is not None else None)
   [Tier 3 boundary — provider constructs per-call AuditedHTTPClient with state_id]
   → On provider error (retryable): RAISE RetrievalError (engine retries)
   → On provider error (non-retryable): TransformResult.error(retryable=False) (quarantine)
6. Check result count
   → Zero results + on_no_results="quarantine": TransformResult.error(retryable=False)
   → Zero results + on_no_results="continue": attach empty-context fields, continue
7. Format context (numbered / separated / raw)
8. Apply max_context_length truncation (if configured)
9. Attach prefixed output fields to row as PipelineRow
10. Return TransformResult.success(output_row, success_reason=...)
```

**Error handling note:** The engine's `_execute_transform_with_retry` retries **only on raised exceptions**, NOT on `TransformResult.error()` — regardless of the `retryable` flag. `LLMTransform` raises `LLMClientError` for retryable failures; the RAG transform follows the same pattern with `RetrievalError`:

- **Retryable failures** (HTTP 429, 5xx, connection timeout): The provider raises `RetrievalError` (see Exception Hierarchy below). The engine catches this, checks the `retryable` attribute, and invokes `RetryManager` for backoff/retry.
- **Non-retryable failures** (HTTP 400, 401, 403, 404, DNS failure): Return `TransformResult.error(retryable=False)` — the row is quarantined immediately.
- **Processing failures** (no regex match, template error, zero results): Return `TransformResult.error(retryable=False)` — these are data-quality or config issues, not transient failures.

### Exception Hierarchy

```python
class RetrievalError(Exception):
    """Base exception for retrieval provider errors."""

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
```

The provider raises `RetrievalError(retryable=True)` for transient failures and `RetrievalError(retryable=False)` for permanent failures. The `status_code` attribute is included for audit traceability — the engine records it in the retry/quarantine context. This mirrors the `WebScrapeError` pattern at `plugins/transforms/web_scrape_errors.py`.

**Processor integration (REQUIRED — engine code change):** The engine's `_execute_transform_with_retry` in `engine/processor.py` currently catches `LLMClientError` and `(ConnectionError, TimeoutError, OSError, CapacityError)` for retry dispatch. **`WebScrapeError` is NOT currently caught** — this is an existing bug where retryable `WebScrapeError` exceptions escape the processor and crash the run.

The correct fix is to introduce a `PluginRetryableError` base class in `contracts/errors.py` (L0) that all plugin retryable exceptions inherit from:

```python
# contracts/errors.py (L0 — no upward dependencies)
class PluginRetryableError(Exception):
    """Base for plugin exceptions eligible for engine retry."""
    retryable: bool
    status_code: int | None
```

Then `RetrievalError`, `WebScrapeError`, and `LLMClientError` all inherit from `PluginRetryableError`. The processor adds `PluginRetryableError` to its exception tuple once — all current and future plugin retryable exceptions are covered without per-plugin processor updates.

**Implementation tasks (blocking — standalone preparatory PR, merged and validated BEFORE the RAG transform PR):**

`PluginRetryableError` MUST be implemented as a standalone preparatory PR, merged and validated before the RAG transform PR. This preparatory PR also fixes the existing bug where `WebScrapeError` escapes the processor. The migration must be atomic within the preparatory PR — all inheritance changes and the processor update land together in one commit.

**Merge-order enforcement (REQUIRED):** Add an import-time assertion in `RetrievalError`'s module to verify `PluginRetryableError` exists with the expected interface. This makes the dependency explicit at import time rather than surfacing as a runtime crash when the first retryable search failure occurs:

```python
# In plugins/infrastructure/clients/retrieval/base.py (or wherever RetrievalError is defined)
from elspeth.contracts.errors import PluginRetryableError  # Fails at import if prep PR not merged
# Verify PluginRetryableError has the expected interface by constructing a test instance
_test = PluginRetryableError("verify", retryable=True)
assert _test.retryable is True, "PluginRetryableError missing retryable attribute"
del _test
```

Implementation tasks:

1. Add `PluginRetryableError` to `contracts/errors.py`
2. **Audit all `isinstance(e, LLMClientError)` and `isinstance(e, WebScrapeError)` call sites** in the codebase (processor, retry manager, executors). Document that re-parenting does not change dispatch behavior — existing `isinstance` checks on the specific subtypes still match, and the new `PluginRetryableError` base class catch in the processor is additive, not a replacement of subtype-specific dispatch elsewhere.
3. Re-parent `LLMClientError` to inherit from `PluginRetryableError`
4. Re-parent `WebScrapeError` to inherit from `PluginRetryableError` (fixes existing bug where retryable `WebScrapeError` escapes the processor and crashes the run)
5. Update `_execute_transform_with_retry` in `engine/processor.py` to catch `PluginRetryableError` instead of listing individual exception types
6. Add regression tests for all three exception types (`LLMClientError`, `WebScrapeError`, `RetrievalError`) verifying retry and quarantine paths. Test file: `tests/unit/engine/test_processor_retry.py` (or add to existing `tests/unit/engine/test_processor.py`). Run: `.venv/bin/python -m pytest tests/unit/engine/test_processor_retry.py -v`

Then in the RAG transform PR:
7. Make `RetrievalError` inherit from `PluginRetryableError`
8. Add integration test: verify retry on `RetrievalError(retryable=True)` from mock provider
9. Add integration test: verify quarantine on `RetrievalError(retryable=False)`

### Output Fields

All output fields are prefixed with the mandatory `output_prefix`:

| Field | Type | Content |
|-------|------|---------|
| `{prefix}__rag_context` | `str` | Formatted retrieved text (the field the LLM template references) |
| `{prefix}__rag_score` | `float` | Top result's relevance score |
| `{prefix}__rag_count` | `int` | Number of chunks retrieved above threshold |
| `{prefix}__rag_sources` | `str` | JSON-serialized `{"v": 1, "sources": [{"source_id": ..., "score": ..., "metadata": ...}, ...]}` |

The transform declares these as `declared_output_fields` for the engine's field collision detection.

**Prefix collision timing:** If two RAG transforms in the same pipeline use the same `output_prefix`, the collision is detected at first-row execution time by the engine's `TransformExecutor` (which checks `declared_output_fields` pre-execution), NOT at config-load or DAG-construction time. This means the run starts, processes zero rows successfully, and fails on the first row. A DAG-construction-time check for `declared_output_fields` overlap across all transforms would surface this earlier — this is an engine-level enhancement to pursue separately. For now, operators must ensure unique prefixes manually.

**Zero-results field values** (when `on_no_results="continue"`): All four fields are still attached to the row with sentinel values:
- `{prefix}__rag_context` = `""` (empty string)
- `{prefix}__rag_score` = `0.0`
- `{prefix}__rag_count` = `0`
- `{prefix}__rag_sources` = `'{"v": 1, "sources": []}'` (versioned empty sources)

This ensures the output schema is consistent regardless of result count — downstream transforms always see the same fields. **Silent semantic degradation risk:** A downstream LLM template referencing `{{ row.policy__rag_context }}` will receive an empty string. The LLM will produce plausible-looking output with no factual grounding. These rows complete as `COMPLETED` in the Landscape — indistinguishable from well-grounded rows in downstream results without querying `success_reason_json` for `retrieval_status: "empty"`. Pipeline architects MUST add a gate between retrieval and generation that routes on `{prefix}__rag_count == 0` when using `on_no_results: continue` with a downstream LLM.

### Context Formatting

Three modes for joining multiple retrieved chunks into `{prefix}__rag_context`:

**Numbered** (default):
```
1. First retrieved chunk text here...
2. Second retrieved chunk text here...
3. Third retrieved chunk text here...
```

**Separated** (with configurable separator):
```
First retrieved chunk text here...
---
Second retrieved chunk text here...
---
Third retrieved chunk text here...
```

**Raw** (simple concatenation):
```
First retrieved chunk text here...Second retrieved chunk text here...Third retrieved chunk text here...
```

### Max Context Length

When `max_context_length` is set, the formatted context is truncated to that character limit. Truncation happens at chunk boundaries where possible — whole chunks are included or excluded, not mid-sentence cuts. If even the first chunk exceeds the limit, it is hard-truncated with a `[truncated]` indicator.

**Storage sizing note:** `max_context_length` also bounds per-row audit data volume. Retrieved context is stored in the row's output data (`node_states.output_data_json`). With `top_k=5` and no `max_context_length`, each row adds approximately `5 × avg_chunk_size` bytes to the audit database. For a 100K-row pipeline at ~1KB per chunk, this is ~500MB of additional audit data. Set `max_context_length` to bound this — it serves double duty as both a token budget tool and a storage hygiene measure.

### Success Reason Metadata

```python
# Imports needed for success_reason metadata:
import hmac
import hashlib
from statistics import mean
from elspeth.contracts.security import get_fingerprint_key

# Variables available from the enclosing process() method:
#   prefix: str = self._config.output_prefix
#   query: str = constructed search query from step 3
#   chunks: list[RetrievalChunk] = results from provider.search() in step 5
#   formatted_context: str = output of context formatting in step 7
#   was_truncated: bool = whether max_context_length truncation was applied in step 8
#   elapsed_ms: float = wall-clock time for the provider.search() call

# output_row is a PipelineRow, not a dict:
# output_row = PipelineRow({**row.to_dict(), **prefixed_fields}, row.contract)
# NOTE: Use row.contract (the input row's contract), NOT ctx.contract.
# ctx.contract is SchemaContract | None — passing None crashes PipelineRow.__init__.
# row.contract is always non-None (the row was constructed with a valid contract).
TransformResult.success(
    output_row,  # PipelineRow (not dict)
    success_reason={
        "action": "enriched",  # Use existing TransformActionCategory literal
        "fields_added": [f"{prefix}__rag_context", f"{prefix}__rag_score",
                         f"{prefix}__rag_count", f"{prefix}__rag_sources"],
        "metadata": {
            "provider": "azure_search",
            "search_mode": self._validated_provider_config.search_mode,
            "query_length": len(query),
            "query_hash": hmac.new(get_fingerprint_key(), query.encode(), hashlib.sha256).hexdigest()[:16],
            # NOTE: get_fingerprint_key() is imported from elspeth.contracts.security (or
            # elspeth.core.security). It is a standalone module-level function, NOT an attribute
            # on TransformContext. TransformContext has no fingerprint_key attribute.
            "chunks_retrieved": len(chunks),
            "top_score": chunks[0].score if chunks else None,
            "mean_score": mean([c.score for c in chunks]) if chunks else None,
            "context_length": len(formatted_context),
            "truncated": was_truncated,
            "retrieval_status": "full" if chunks else "empty",  # Enables Landscape queries for degraded rows
            "latency_ms": round(elapsed_ms),
        },
    },
)
```

Note: Provider-specific fields are nested under `metadata` (a `NotRequired[dict[str, Any]]` on `TransformSuccessReason`), not placed at the top level. The `action` uses the existing `"enriched"` literal from `TransformActionCategory`. `search_mode` is recorded for audit traceability — different modes produce different score distributions, and an auditor must be able to determine which normalization was in effect. `latency_ms` is included for consistency with the LLM transform's success reason.

The `retrieval_status` field distinguishes `"full"` (chunks returned) from `"empty"` (zero results with `on_no_results: continue`). This enables Landscape queries to identify rows that proceeded with incomplete context — e.g., `SELECT * FROM node_states WHERE success_reason_json->>'retrieval_status' = 'empty'`.

This metadata is recorded in `node_states.success_reason_json` in the Landscape — fully auditable.

### Error Reason Metadata

Each error path produces a distinct reason dict for audit traceability. Error reason strings must be members of `TransformErrorCategory` (a closed `Literal[...]` type in `contracts/errors.py`). Where existing categories apply, they are used directly. New categories must be added to `TransformErrorCategory` before implementation:

```python
# Regex no-match — NEW: add "no_regex_match" to TransformErrorCategory
TransformResult.error(
    {"reason": "no_regex_match", "field": query_field, "pattern": query_pattern},
    retryable=False,
)

# Template render error — EXISTING: "template_rendering_failed" already in TransformErrorCategory
TransformResult.error(
    {"reason": "template_rendering_failed", "error": str(e), "field": query_field},
    retryable=False,
)

# Provider error (retryable — HTTP 429, 5xx, connection timeout)
# RAISED as exception — engine retries via RetryManager
# NEW: add "retrieval_failed" to TransformErrorCategory
raise RetrievalError(
    f"Retrieval failed ({status_code}): {error_message}",
    retryable=True,
    status_code=status_code,
)
# The engine records the exception context in node_states for the failed attempt,
# then retries. If retries exhaust, the row is quarantined with the last error.

# Provider error (non-retryable — HTTP 400, 401, 403, 404, DNS failure)
TransformResult.error(
    {"reason": "retrieval_failed", "provider": provider_name, "error": str(e),
     "status_code": status_code, "query_length": len(query)},
    retryable=False,
)

# Zero results — NEW: add "no_results" to TransformErrorCategory
TransformResult.error(
    {"reason": "no_results", "provider": provider_name, "query_length": len(query), "min_score": min_score},
    retryable=False,
)
```

**Retryable classification:** HTTP 429 (rate limited) and 5xx (server error) raise `RetrievalError(retryable=True)` — the engine retries. HTTP 400 (bad request), 403 (forbidden), 404 (not found) return `TransformResult.error(retryable=False)` — the row is quarantined. Connection timeouts raise `RetrievalError(retryable=True)`. DNS resolution failures return `TransformResult.error(retryable=False)`.

**HTTP 401 classification — retryable once:** A 401 from Azure AI Search may indicate a transient token expiry (managed identity tokens expire and `azure-identity` handles transparent refresh, but the refresh can fail transiently). Classify HTTP 401 as `RetrievalError(retryable=True)` with a recommendation that `max_attempts` for this transform be kept low (2-3). If the retry also returns 401, the engine exhausts retries and quarantines the row. This matches the Dataverse spec's 401 handling pattern (credential reconstruction + single retry).

**`TransformErrorCategory` additions required (explicit implementation prerequisite):** The three new literals — `"no_regex_match"`, `"retrieval_failed"`, `"no_results"` — plus `"invalid_input"` MUST be added to the `TransformErrorCategory` literal in `contracts/errors.py` before the transform can be implemented. These should be part of the `PluginRetryableError` preparatory PR or a separate prerequisite commit. The existing `"template_rendering_failed"` covers query template render errors. `"invalid_input"` covers the None-value and empty/whitespace query validation paths added in the process flow (steps 2 and 4).

### Note on `{prefix}__rag_sources` Field Type

The `__rag_sources` field is stored as a JSON-serialized string because `PipelineRow` field values are restricted to contract types (str, int, float, bool, None, datetime, object). Storing a `list[dict]` directly would violate the row contract — complex nested structures are not contract types. This is a deliberate constraint — the field's primary consumer is the Landscape audit trail (where it's recorded as-is in `success_reason_json`), not downstream transforms.

**Version marker:** The `__rag_sources` field uses a versioned envelope format (`{"v": 1, "sources": [...]}`) rather than a bare list. The version marker (`v: 1`) enables schema evolution — downstream consumers and audit tooling can distinguish format versions without migration. Adding versioning retroactively after data is in production audit tables is costly.

**Downstream consumption rules:** If a downstream transform or sink needs to parse `__rag_sources`, the parse is on data we serialized — this is Tier 1 data (we wrote it, we own it). Per CLAUDE.md's tier model: parse errors indicate a bug in the RAG transform (corruption), not a data quality issue. Downstream consumers should use `json.loads()` **without** try/except — if the parse fails, that's a crash-worthy bug to fix, not an error to handle gracefully.

## Pipeline Composition Examples

### Single retrieval + LLM

```yaml
transforms:
  - plugin: rag_retrieval
    options:
      output_prefix: "policy"
      query_field: "customer_question"
      provider: azure_search
      provider_config:
        endpoint: "https://my-search.search.windows.net"
        index: "policy-documents"
        api_key: "${AZURE_SEARCH_API_KEY}"
        search_mode: hybrid
      top_k: 5
      min_score: 0.7
      on_no_results: quarantine
      context_format: numbered
      max_context_length: 4000
      schema:
        mode: observed

  - plugin: llm
    options:
      template: |
        Answer the customer's question using the provided policy context.

        Question: {{ row.customer_question }}

        Relevant policies:
        {{ row.policy__rag_context }}

        Answer:
```

### Multi-source retrieval + LLM

```yaml
transforms:
  - plugin: rag_retrieval
    options:
      output_prefix: "financial"
      query_field: "complaint_text"
      provider: azure_search
      provider_config:
        endpoint: "https://my-search.search.windows.net"
        index: "financial-policies"
        api_key: "${AZURE_SEARCH_API_KEY}"
      top_k: 5
      min_score: 0.7
      schema:
        mode: observed

  - plugin: rag_retrieval
    options:
      output_prefix: "regulatory"
      query_field: "product_category"
      query_template: "Regulations for product type: {{ query }}"
      provider: azure_search
      provider_config:
        endpoint: "https://my-search.search.windows.net"
        index: "regulatory-docs"
        api_key: "${AZURE_SEARCH_API_KEY}"
      top_k: 3
      min_score: 0.8
      schema:
        mode: observed

  - plugin: llm
    options:
      template: |
        Classify the customer complaint considering both financial policies
        and regulatory requirements.

        Complaint: {{ row.complaint_text }}

        Financial policies:
        {{ row.financial__rag_context }}

        Regulatory context:
        {{ row.regulatory__rag_context }}

        Classification:
```

### Retrieval with regex extraction

```yaml
transforms:
  - plugin: rag_retrieval
    options:
      output_prefix: "context"
      query_field: "raw_email_body"
      query_pattern: "(?:issue|problem|complaint):\\s*(.+?)(?:\\n|$)"
      provider: azure_search
      provider_config:
        endpoint: "https://my-search.search.windows.net"
        index: "knowledge-base"
        api_key: "${AZURE_SEARCH_API_KEY}"
      top_k: 3
      on_no_results: continue  # LLM can try without context
      schema:
        mode: observed
```

## Dependencies

| Dependency | Extra | Purpose |
|-----------|-------|---------|
| `httpx` | base | HTTP client (via `AuditedHTTPClient`) |
| `azure-identity` | `[azure]` | Managed identity auth for Azure AI Search |

No new dependencies required for the day-one implementation. Azure AI Search is a REST API — no dedicated SDK needed.

When non-Azure providers are added later, their dependencies would go into a new `[rag]` extra (e.g., `chromadb`, `pinecone-client`). The `[azure]` extra remains sufficient for the Azure AI Search provider.

## Plugin Discovery

The transform lives in `plugins/transforms/rag/`. The plugin scanner's `PLUGIN_SCAN_CONFIG` must be updated to include `transforms/rag` as a scanned subdirectory (similar to `transforms/llm` and `transforms/azure`).

**Explicit implementation task:** Update `PLUGIN_SCAN_CONFIG` in `plugins/infrastructure/discovery.py` to include the `plugins/transforms/rag/` directory. Without this, `discover_all_plugins()` will not find `RAGRetrievalTransform`.

The retrieval client infrastructure (`plugins/infrastructure/clients/retrieval/`) does not need scanner registration — it's imported directly by the transform, not discovered.

## Testing Strategy

### Test File Locations

```
tests/unit/plugins/transforms/rag/test_query.py
tests/unit/plugins/transforms/rag/test_formatter.py
tests/unit/plugins/transforms/rag/test_config.py
tests/unit/plugins/transforms/rag/test_transform.py
tests/unit/plugins/infrastructure/clients/retrieval/test_azure_search.py
tests/unit/plugins/infrastructure/clients/retrieval/test_types.py
tests/integration/plugins/transforms/test_rag_pipeline.py
```

**Run commands:**
```bash
.venv/bin/python -m pytest tests/unit/plugins/transforms/rag/ -v
.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/ -v
.venv/bin/python -m pytest tests/integration/plugins/transforms/test_rag_pipeline.py -v
```

### Unit Tests

**Query construction (`test_query.py`):**
- Field-only mode: extracts value verbatim
- Field-only mode: None value, empty string, whitespace-only
- Template mode: renders query with `{{ query }}` and `{{ row }}` context
- Template mode: structural error at compile time (not row time)
- Template mode: render error produces `TransformResult.error()`
- Regex mode: captures first group
- Regex mode: uses full match when no capture groups
- Regex mode: no match produces `TransformResult.error()`
- Regex mode: timeout on pathological backtracking pattern produces `TransformResult.error()`
- Regex mode: invalid pattern rejected at config validation

**Context formatting (`test_formatter.py`):**
- Numbered format with multiple chunks
- Numbered format with single chunk
- Separated format with custom separator
- Raw format (concatenation)
- Max length truncation at chunk boundaries
- Max length truncation mid-chunk (with `[truncated]` indicator)
- `max_context_length=0` rejected by `ge=1` validation
- Empty chunk list handling (zero-results-continue path)

**Config validation (`test_config.py`):**
- `output_prefix` must be valid Python identifier
- `output_prefix` rejects Python keywords
- `query_template` and `query_pattern` mutually exclusive
- `query_pattern` regex compilation validation
- `top_k` bounds (1-100)
- `min_score` bounds (0.0-1.0)
- Provider config eagerly validated

**Provider types (`test_types.py`):**
- `RetrievalChunk` score boundary values (0.0, 1.0, out of range)
- `RetrievalChunk` metadata JSON-serializability (datetime rejected, str accepted)
- `RetrievalChunk.__post_init__` error messages

**Provider protocol (`test_azure_search.py`):**
- Azure search config validation (auth mutual exclusion, semantic config requirement)
- Score normalization per search mode (keyword, vector, hybrid, semantic)
- Tier 3 boundary: malformed JSON response
- Tier 3 boundary: missing `value` array
- Tier 3 boundary: missing content field in result items
- Tier 3 boundary: metadata coercion (datetime → ISO 8601 string)
- Per-call AuditedHTTPClient construction with correct state_id

**Transform lifecycle and process flow (`test_transform.py`):**
- `process()` with field-only query mode: correct output fields attached
- `process()` with template query mode: template rendered with row context
- `process()` with regex query mode: first capture group used as query
- `on_no_results: quarantine` path: returns `TransformResult.error()` with correct category
- `on_no_results: continue` path: returns `TransformResult.success()` with empty sentinel fields
- Retryable error dispatch: `RetrievalError(retryable=True)` propagates (not caught by transform)
- Non-retryable error dispatch: `RetrievalError(retryable=False)` returns `TransformResult.error()`
- `success_reason` metadata structure: all expected fields present, `query_hash` uses HMAC
- `on_start()` stores lifecycle context correctly (landscape, run_id, telemetry_emit, rate_limit_registry)
- `on_complete()` emits statistics via `telemetry_emit`, guards `mean()` for zero-row case
- `close()` before `on_start()`: does not raise (guard on `_provider is None`)
- `close()` after normal lifecycle: calls `provider.close()`

### Integration Tests

- Full transform process: query → retrieve → format → output (mock HTTP transport)
- Zero results with `on_no_results: quarantine` → `TransformResult.error()`
- Zero results with `on_no_results: continue` → empty context fields with sentinel values
- Multiple chunks → correct output field population
- Output field prefix applied correctly
- Declared output fields match actual output (collision detection compatibility)
- Per-row audit recording: verify each row's search call is recorded under its own state_id
- Pipeline composition: RAG transform → LLM transform (mock both external calls)
- Plugin discovery: `discover_all_plugins()` finds `rag_retrieval` after `PLUGIN_SCAN_CONFIG` update
- `on_complete()` with zero rows processed (no StatisticsError)

### Tier Model Compliance

Tests use production code paths. The `AzureSearchProvider` is tested via mock HTTP transport on `AuditedHTTPClient`, not by mocking the provider itself. Integration tests use `instantiate_plugins_from_config()` for plugin construction.

## Security Considerations

- **API key handling:** Azure AI Search API keys are resolved via ELSPETH's secret management. HMAC fingerprints in audit trail, never raw keys.
- **Query injection:** OData/search query construction does not use string interpolation of user data into filter expressions. The search query is passed as a request body parameter, not interpolated into a URL or filter string.
- **Template sandbox:** Query templates use `ImmutableSandboxedEnvironment` with `StrictUndefined` from `plugins/infrastructure/templates.py` (shared infra, same sandbox used by LLM prompt templates). This prevents attribute access, method calls, and module imports in user-provided templates. **Limitation:** The sandbox does not limit CPU or memory consumption from template loops — acceptable since templates are authored by pipeline architects (trusted config), not end users.
- **Context size:** `max_context_length` is a **token budget management** feature — it caps the character length of retrieved context to prevent oversized payloads. It does NOT defend against prompt injection. A malicious document in the search index that contains instruction-override text (e.g., "Ignore all previous instructions...") will pass through regardless of length.
- **Prompt injection via retrieved content:** Prompt injection defense is **out of scope for the retrieval transform**. The retrieval transform fetches and formats context; it does not interpret or execute it. Defense against adversarial content in retrieved documents belongs in either (a) a content safety gate between retrieval and generation (e.g., Azure Content Safety transform), or (b) the LLM transform's prompt design (instruction separation techniques). Pipeline architects should add a safety gate when the search index contains user-generated or untrusted content.
- **Source attribution:** `{prefix}__rag_sources` provides full traceability from LLM output back to the specific documents that informed it — critical for audit.

## Future Extensibility

The `RetrievalProvider` protocol is the extensibility boundary. Adding a new backend requires:

1. Implement `RetrievalProvider` (search + close methods)
2. Add a config model for the provider
3. Register the provider name in the transform's provider dispatch

No changes to the transform, query construction, or formatting code. This is the lesson learned from the LLM transform refactor — the protocol boundary prevents vendor-specific code from becoming load-bearing in the core transform logic.

When non-Azure providers are added, the `[azure]` extra constraint relaxes to a new `[rag]` extra that includes the appropriate client libraries.
