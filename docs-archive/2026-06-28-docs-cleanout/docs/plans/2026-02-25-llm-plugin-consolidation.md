# T10: LLM Plugin Consolidation Design

**Date:** 2026-02-25
**Status:** Approved
**Task:** elspeth-rapid-a6bde1
**Branch:** RC3.3-architectural-remediation

## Problem

6 separate LLM transform classes across 6 files total ~4,950 lines with severe
duplication. Two providers (Azure, OpenRouter) x three modes (single-query,
multi-query, batch) are implemented as independent classes that drift apart as
bugs are fixed in one but not others.

**Duplication inventory:**

| Category | Lines | Files |
|----------|-------|-------|
| Langfuse tracing (3 methods x 6 files) | ~600 | All 6 |
| Response parsing/validation | ~100 | OpenRouter variants |
| Template rendering error handling | ~32 | All 4 non-batch |
| Truncation detection | ~50 | Both multi-query |
| Markdown fence stripping | ~20 | Both multi-query |
| Output row construction | ~30 | Both single-query |
| **Total** | **~830** | |

Additional structural problems:
- Domain-specific terminology (`case_studies x criteria`) baked into multi-query
- Diamond config inheritance (`MultiQueryConfig(AzureOpenAIConfig, MultiQueryConfigMixin)`)
- Single-query and multi-query are separate code paths despite sharing infrastructure
- Adding a new provider requires implementing 2-3 full classes

## Solution: Strategy Pattern with Provider Protocol

### Core Pattern

A unified `LLMTransform` class delegates provider-specific transport to an
`LLMProvider` strategy object. Two internal processing strategies
(`SingleQueryStrategy`, `MultiQueryStrategy`) share the same provider,
tracing, and lifecycle infrastructure but have distinct processing logic.

```text
LLMTransform
├── LLMProvider protocol (transport layer)
│   ├── AzureLLMProvider (~120 lines)
│   └── OpenRouterLLMProvider (~300 lines)
├── Processing strategies (processing layer)
│   ├── SingleQueryStrategy (~100 lines)
│   └── MultiQueryStrategy (~100 lines)
├── LangfuseTracer (extracted tracing)
└── QuerySpec[] (domain-agnostic query model)
```

### Architecture Decisions

**D1: Two strategies, not one code path.**
Single-query and multi-query have genuinely different processing models:
- Single-query: template renders against raw row, output is `{response_field: content}`,
  contract uses `propagate_contract()`
- Multi-query: template renders against synthetic context (mapped fields), output is
  `{prefix_suffix: value}` per output mapping, contract rebuilt as OBSERVED

Forcing single-query through multi-query's path would require fake wrapper values or
hidden `if single_query:` branches — worse than explicit strategies. The strategies
are small (~100 lines each). The shared infrastructure eliminates the duplication.

**D2: Providers own audit recording.**
Each provider holds its own `Audited*Client` instance. Landscape call recording
happens inside `execute_query()`, matching the existing trust boundary pattern.
The transform never sees raw SDK/HTTP responses — only validated `LLMQueryResult`.

**D3: Provider lifecycle is per-state_id.**
Providers manage client caching with thread-safe locking, matching the existing
per-state_id pattern. The protocol does not expose this detail — it is internal
to each provider implementation.

**D4: Tracing belongs on the transform, not the provider.**
`setup_tracing()` is removed from the provider protocol. Azure AI auto-instrumentation
(which hooks the SDK) is set up during provider construction as a side effect.
Langfuse tracing is managed by `LangfuseTracer` at the transform level.

**D5: No query_groups (YAGNI).**
The arbitrary N-dimensional cross-product expansion was cut. Explicit `queries`
(list or dict) covers all current use cases. The existing two-dimensional
cross-product for evaluation matrices can be retained with renamed fields if
needed in a future task.

**D6: Keep batch transforms under llm/.**
Splitting into `llm_batch/` creates a false separation — batch transforms share
config, templates, tracing, and metadata utilities. A directory boundary makes
future consolidation psychologically harder without providing real isolation.

**D7: Two-phase implementation.**
Phase A extracts shared infrastructure (independently committable). Phase B
introduces the provider protocol and unified transform (builds on stable shared code).
This avoids Big Bang risk on a ~4,950-line refactoring.

**D8: LLMTransform retains BatchTransformMixin.**
All 6 existing LLM transforms use `BatchTransformMixin` with `accept()`/
`connect_output()`/`flush_batch_processing()` — NOT `process()`. The existing
`azure.py` and `openrouter.py` explicitly `raise NotImplementedError` on
`process()`. The unified `LLMTransform` extends `BatchTransformMixin` and
strategies are called from `_process_row()`, preserving concurrent row
processing with FIFO output ordering and backpressure. The engine executor
has separate code paths for `BatchTransformMixin` transforms — dropping it
would be a silent performance regression.

**D9: MultiQueryStrategy traces per-query only (Azure behavior).**
The existing `azure_multi_query.py` overrides `_record_row_langfuse_trace` to
a deliberate no-op — Azure traces per-query, not per-row. The existing
`openrouter_multi_query.py` uses the base class row-level aggregate trace.
The unified `MultiQueryStrategy` traces per-query only (the more granular
approach). Per-query traces are always emitted; row-level aggregate traces
are dropped. This simplifies the tracing model and provides better Langfuse
visibility. The `system_prompt` is shared across all queries at the transform
level, not per-query.

### Component Structure

```text
plugins/llm/
├── __init__.py              # Keep — shared field helpers (already good)
├── base.py                  # MODIFY — unified LLMConfig, flat hierarchy
├── validation.py            # MODIFY — expand with shared validation functions
├── tracing.py               # Keep — tracing config models
├── templates.py             # Keep — PromptTemplate
├── multi_query.py           # MODIFY — domain-agnostic QuerySpec, drop case_studies/criteria
│
├── provider.py              # NEW — LLMProvider protocol + LLMQueryResult DTO
├── providers/
│   ├── __init__.py
│   ├── azure.py             # NEW — AzureLLMProvider
│   └── openrouter.py        # NEW — OpenRouterLLMProvider
├── transform.py             # NEW — LLMTransform + SingleQueryStrategy + MultiQueryStrategy
├── langfuse.py              # NEW — extracted LangfuseTracer
│
├── azure.py                 # DELETE (replaced by providers/azure.py + transform.py)
├── openrouter.py            # DELETE (replaced by providers/openrouter.py + transform.py)
├── base_multi_query.py      # DELETE (absorbed into transform.py strategies)
├── azure_multi_query.py     # DELETE (absorbed into transform.py)
├── openrouter_multi_query.py # DELETE (absorbed into transform.py)
│
├── azure_batch.py           # KEEP (different execution model, adopts shared infra)
└── openrouter_batch.py      # KEEP (different execution model, adopts shared infra)
```

### Provider Protocol

```python
@dataclass(frozen=True, slots=True)
class LLMQueryResult:
    """Normalized, validated result from any LLM provider."""
    content: str                          # Validated, non-null, non-empty
    usage: TokenUsage                     # Normalized via TokenUsage.known/unknown
    model: str                            # Actual responding model
    finish_reason: FinishReason | None = None  # Validated enum, not raw string

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("LLMQueryResult.content must be non-empty (whitespace-only rejected)")
        if not self.model:
            raise ValueError("LLMQueryResult.model must be non-empty")

class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"

class LLMProvider(Protocol):
    """Narrow interface — transport only. 2 methods."""
    def execute_query(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
        state_id: str,
        token_id: str,
    ) -> LLMQueryResult: ...

    def close(self) -> None: ...
```

Providers raise typed exceptions from `elspeth.plugins.clients.llm`:
`RateLimitError`, `ContentPolicyError`, `NetworkError`, `ServerError`,
`LLMClientError` (base class for non-categorized failures). The transform
converts these to `TransformResult.error` or re-raises for RetryManager.

**Note:** `raw_response` is NOT on `LLMQueryResult`. Providers own audit
recording (D2) — the raw SDK/HTTP response is recorded to the Landscape
inside `execute_query()` via `AuditedLLMClient`/`AuditedHTTPClient` and
never travels to the transform layer. All data the transform needs
(`content`, `usage`, `model`, `finish_reason`) is extracted and typed at
the provider boundary.

**Finish reason normalization:** Each provider normalizes raw finish_reason
strings at the Tier 3 boundary before constructing `LLMQueryResult`.
Provider-specific aliases (e.g. `"end_turn"`, `"max_tokens"`, `"COMPLETE"`)
are mapped to `FinishReason` enum values or `None` with a warning log.
This keeps normalization inside the provider where the provider-specific
knowledge lives.

### Query Model (Domain-Agnostic)

```python
@dataclass(frozen=True, slots=True)
class QuerySpec:
    """One query to execute against an LLM for a given row."""
    name: str                          # Output field prefix, template key
    input_fields: dict[str, str]       # {template_var: row_field}
    response_format: ResponseFormat
    output_fields: list[OutputFieldConfig] | None = None
    template: str | None = None        # Per-query override
    max_tokens: int | None = None      # Per-query override

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("QuerySpec.name must be non-empty")
        if not self.input_fields:
            raise ValueError("QuerySpec.input_fields must be non-empty")
```

**`resolve_queries()` must validate:**
- Empty `queries: []` raises `PluginConfigError` (not silently zero queries)
- Output field name collisions (ports logic from `validate_multi_query_key_collisions`)
- Reserved suffix collisions with `LLM_AUDIT_SUFFIXES` / `LLM_GUARANTEED_SUFFIXES`

Config accepts list or dict forms (dict keys become names):

```yaml
# Single query (implicit — omit queries entirely)
transforms:
  - plugin: llm
    provider: azure
    options:
      model: gpt-4
      template: "Classify: {{ text }}"
      response_field: classification

# Multi-query — dict-keyed for readable names
transforms:
  - plugin: llm
    provider: openrouter
    options:
      model: anthropic/claude-sonnet
      template: "Evaluate {{ text }} for {{ dimension }}"
      queries:
        sentiment:
          input_fields: {text: feedback, dimension: "sentiment"}
        toxicity:
          input_fields: {text: feedback, dimension: "toxicity"}
```

### Config Model (Flat Hierarchy)

**Schema change:** `model` changes from required (`str = Field(...)`) to
optional (`str | None = None`). This affects all test fixtures that rely on
`model` being required. Azure's `AzureOpenAIConfig` defaults `model` to
`deployment_name` via validator, so omitting it is valid. OpenRouter requires
it — enforced via a Pydantic validator on `OpenRouterConfig`.

```python
class LLMConfig(TransformDataConfig):
    provider: Literal["azure", "openrouter"]  # Required — validated at config-load time
    model: str | None = None              # None → provider-specific default
    template: str | TemplatePath = ...
    temperature: float = 0.0
    max_tokens: int | None = None
    response_field: str = "llm_response"  # Single-query output field
    response_format: ResponseFormat = ResponseFormat.STANDARD
    queries: list[QuerySpec] | dict[str, QuerySpecBody] | None = None
    tracing: TracingConfig | None = None
    max_concurrent_queries: int = 3

class AzureOpenAIConfig(LLMConfig):
    deployment_name: str
    endpoint: str
    api_key: str
    api_version: str = "2024-10-21"

class OpenRouterConfig(LLMConfig):
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 60.0
```

No more `MultiQueryConfigMixin` or diamond inheritance. Multi-query fields
live in base `LLMConfig`. Pydantic validator normalizes dict→list for queries.

### Plugin Registration

5 plugin names collapse to 1:

| Before | After |
|--------|-------|
| `azure_llm` | `llm` with `provider: azure` |
| `openrouter_llm` | `llm` with `provider: openrouter` |
| `azure_multi_query_llm` | `llm` with `provider: azure` + `queries:` |
| `openrouter_multi_query_llm` | `llm` with `provider: openrouter` + `queries:` |
| `azure_batch_llm` | Unchanged at the time. *(Retired 2026-05-06 per ADR-020.)* |
| `openrouter_batch_llm` | Unchanged at the time. *(Retired 2026-05-06 per ADR-020.)* |

Provider routing is internal to `LLMTransform.__init__` via dict lookup.
Not exposed through pluggy. Old plugin names get a helpful error message,
no compatibility shim (no-legacy-code policy).

**Config validation:** `_get_transform_config_model()` reads the `provider`
field from the raw config dict and returns the provider-specific config class
(`AzureOpenAIConfig` or `OpenRouterConfig`). This ensures `elspeth validate`
catches missing provider-specific fields (e.g. `deployment_name` for Azure)
at config-load time, not at runtime instantiation.

### Transform Lifecycle

`LLMTransform` extends `BatchTransformMixin` (D8). Rows arrive via
`accept()` → internal queue → `_process_row()`, preserving concurrent
row processing with FIFO output ordering.

```text
Row arrives at LLMTransform._process_row() (via BatchTransformMixin.accept())
│
├─ Delegate to strategy (SingleQueryStrategy or MultiQueryStrategy)
│
│  SingleQueryStrategy:
│  ├─ Render template against raw row
│  ├─ provider.execute_query() → LLMQueryResult
│  ├─ Check truncation (finish_reason == LENGTH)
│  ├─ Strip markdown fences if STANDARD format
│  ├─ output[response_field] = content
│  ├─ populate_llm_metadata_fields()
│  ├─ propagate_contract()
│  └─ Return TransformResult.success
│
│  MultiQueryStrategy:
│  ├─ For each QuerySpec: build template context, render template
│  ├─ Fan out via PooledExecutor → list[LLMQueryResult]
│  ├─ For each result: truncation check, fence strip, JSON parse, field extract
│  ├─ Merge all query outputs into row
│  ├─ Build OBSERVED SchemaContract
│  └─ Return TransformResult.success
│
└─ Error handling:
   ├─ Template failure → TransformResult.error (not retryable)
   ├─ RateLimitError → re-raise (RetryManager)
   ├─ NetworkError/ServerError → re-raise (retryable)
   ├─ ContentPolicyError → TransformResult.error (not retryable)
   ├─ Truncation → TransformResult.error (not retryable)
   └─ JSON validation failure → TransformResult.error (not retryable)
```

Partial multi-query failure = full row failure (audit integrity).

### Langfuse Extraction

Uses factory pattern to avoid mutable two-phase initialization:

```python
def create_langfuse_tracer(
    transform_name: str,
    tracing_config: TracingConfig | None,
) -> LangfuseTracer:
    """Returns ActiveLangfuseTracer or NoOpLangfuseTracer."""

class LangfuseTracer(Protocol):
    """What the transform needs. Narrow interface."""
    def record_success(self, token_id, query_name, prompt, response_content, model, ...): ...
    def record_error(self, token_id, query_name, prompt, error_message, model, ...): ...
    def flush(self) -> None: ...
```

The transform holds `LangfuseTracer` (always non-None, may be no-op). No
`is_active` checks needed — the no-op implementation silently returns.

**Tracing failures go to structlog only** — not to the ELSPETH telemetry
stream. Langfuse tracing is Tier 2 operational data; failures are logged
at warning level via `structlog`. The `telemetry_emit` callback expects
`ExternalCallCompleted` dataclass instances (from `plugins/clients/base.py`),
which do not match tracing failure events. Simplifying to log-only avoids
a type mismatch and keeps the `LangfuseTracer` protocol clean.

Replaces 3 methods x 6 files (~600 lines) with 2 methods x 1 file (~80 lines).

**`PluginContext.llm_client` disposition:** After T10, the unified `LLMTransform`
does not use `ctx.llm_client` — providers hold their own `Audited*Client`
instances (D2). The executor may still set `ctx.llm_client` (it does not
know the transform's internals), but `LLMTransform._process_row()` MUST NOT
read it. A test (`test_llm_transform_does_not_use_ctx_llm_client`) verifies
this by setting `ctx.llm_client` to a sentinel that raises on access.
Batch transforms continue to use the executor-provided client.
Removing `ctx.llm_client` from the executor path is deferred to T17.

## Implementation Phases

### Phase A: Extract Shared Infrastructure
1. Create `langfuse.py` — extract `LangfuseTracer` from all 6 files
2. Expand `validation.py` — shared template error handling, truncation detection,
   markdown fence stripping
3. Update all 6 existing transforms to use extracted utilities
4. All existing tests pass (no behavioral change)
5. Commit

### Phase B: Provider Protocol + Unified Transform
1. Create `provider.py` — `LLMProvider`, `LLMQueryResult`, `FinishReason`
2. Create `providers/azure.py` — `AzureLLMProvider`
3. Create `providers/openrouter.py` — `OpenRouterLLMProvider`
4. Create `transform.py` — `LLMTransform`, `SingleQueryStrategy`, `MultiQueryStrategy`
5. Refactor `base.py` — flat config hierarchy
6. Refactor `multi_query.py` — domain-agnostic `QuerySpec`
7. Update plugin registration (1 plugin name)
8. Migrate tests (421 class refs across 24 files)
9. Update example YAML files and documentation (16 YAMLs + 8 doc files)
10. Delete old files (`azure.py`, `openrouter.py`, `base_multi_query.py`,
    `azure_multi_query.py`, `openrouter_multi_query.py`)
11. Commit

## Expected Impact

- ~4,200 lines deleted, ~900 lines created = **~3,300 net reduction**
- 5 plugin names → 1
- Diamond config inheritance → flat hierarchy
- Domain-specific terminology → domain-agnostic QuerySpec
- New provider = ~100-300 line class (vs 2-3 full transforms)
- ChaosLLM slots in as test provider via same protocol
- Batch transforms adopt extracted tracing/validation (not restructured)

## Review Panel

### Round 1 (design review, 2026-02-25)

Design reviewed by three specialized agents:

- **Architecture Critic** (4/5): Core strategy pattern correct. Flagged
  single→multi-query forced unification, query_groups YAGNI, audit recording
  layer ambiguity. All addressed in revisions.
- **Systems Thinker** (Leverage Level 10): Confirmed genuine structural leverage
  breaking divergence reinforcing loop. Recommended two-phase implementation,
  two strategies over forced unification, keeping batch under llm/.
- **Python Engineering Reviewer** (3C/5W/3S): Flagged raw_response typing,
  provider lifecycle per-state_id, setup_tracing placement. Endorsed flat
  config hierarchy and LangfuseTracer extraction.

### Round 2 (full peer review, 2026-02-25)

Implementation plan reviewed by 7 specialized agents (architecture, reality-check,
quality/testing, systems/risk, architecture critic, Python code review, type design).
Verdict: **APPROVE WITH CHANGES**. All changes incorporated:

- **B1:** Removed `raw_response` from `LLMQueryResult` — providers own audit
  recording (D2), transform never needs raw response
- **B2:** Fixed `LLMCallError` → `LLMClientError` (correct exception from
  `plugins/clients/llm.py`, not the frozen dataclass in `contracts/call_data.py`)
- **B3:** Removed local `TransformErrorReason = dict[str, Any]` — imports
  existing TypedDict from `contracts/`
- **B4:** Rewrote `LangfuseTracer` as factory + frozen dataclass + no-op pattern
  (eliminates mutable two-phase init, thread-safe for BatchTransformMixin)
- **B5:** Added missing test files to migration inventory (property tests,
  integration tests, corrected paths)
- **B6:** Providers normalize finish_reason at Tier 3 boundary; unknown values
  logged at warning level
- **B7:** Config validation returns provider-specific class via dispatch, not
  base `LLMConfig`
- **H1-H3, M1-M4:** Config schema change documented, key collision prevention
  ported, flush telemetry added, PluginContext.llm_client disposition documented,
  template variable migration documented, empty queries validated, `__post_init__`
  added to DTOs

### Round 3 (full peer review, 2026-02-25)

Design + implementation plan reviewed by 7 specialized agents (architecture,
reality-check, quality/testing, systems/risk, architecture critic, Python
code review, type design). Verdict: **APPROVE WITH CHANGES**. All changes
incorporated:

- **B1:** Added D8 — `LLMTransform` retains `BatchTransformMixin`, strategies
  called from `_process_row()`. Lifecycle diagram updated. Existing transforms
  use `accept()`/`connect_output()`, not `process()`.
- **B2:** Fixed `TelemetryEmitCallback` import path — correct location is
  `elspeth.plugins.clients.base`, not hallucinated `elspeth.contracts.telemetry`
- **B3:** Removed `telemetry_emit` from `LangfuseTracer` protocol — tracing
  failures go to structlog only. `TelemetryEmitCallback` expects
  `ExternalCallCompleted`, not a plain dict.
- **H1:** `NoOpLangfuseTracer` now matches Protocol signatures explicitly
  (no `*args/**kwargs`). Enables mypy to catch signature drift.
- **H2:** Added OpenRouter response parsing helpers to Task 2 scope
  (`parse_llm_json_response`)
- **H3:** `state_id` snapshot bug documented — Azure has fix, OpenRouter
  doesn't. Providers must snapshot `state_id` before try block.
- **H4:** `LLMQueryResult.__post_init__` rejects whitespace-only content
  via `content.strip()` check
- **H5:** Template variable migration verified against `StrictUndefined`
  policy — un-migrated `{{ input_1 }}` raises `TemplateError`, not empty render
- **H6:** Added docs/ to migration scope (33 refs in 8 files) and
  verification checklist
- **H7:** Added D9 — `MultiQueryStrategy` traces per-query only (Azure
  behavior). Row-level aggregate traces dropped.
- **H8:** Phase A Task 3 `openrouter_multi_query.py` tracing alignment
  flagged as behavior change with targeted test requirement

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Big Bang transition (~4,950 lines) | Two-phase: extract shared (A) then restructure (B) |
| Provider-specific edge cases lost | Maintain provider-specific test cases post-consolidation |
| Test blast radius (421 refs, 30+ files) | Explicit test migration in Phase B including property + integration tests |
| Batch transforms diverge further | Adopt extracted langfuse.py + validation.py in same pass |
| Example YAML breakage | Integration test validates all examples parse |
| Multi-query template variable change | Explicit before/after migration for `{{ input_N }}` → named vars; verify `PromptTemplate` uses `StrictUndefined` so missing vars raise `TemplateError` rather than rendering empty |
| Config validation regression | `Literal["azure", "openrouter"]` on `provider` field + provider-dispatch in `_get_transform_config_model()` |
| BUG-LINEAGE-01 risk in fixture migration | Strategy-type assertions in migrated multi-query tests |
| `state_id` snapshot bug (Azure-only fix) | `azure.py:453` snapshots `state_id` before try block because `ctx.state_id` is mutable during retries; `openrouter.py:693` uses `ctx.state_id` directly (buggy). Unified provider MUST use the Azure pattern (snapshot). |
| Doc references to old plugin names | 33 refs in 8 doc files (`tier2-tracing.md`, `user-manual.md`, etc.) — update in Phase B alongside YAML migration |

## Sequencing

Execute T10 before T17 (PluginContext split) and T19 (Landscape repos) to avoid
concurrent changes to `LLMTransform` and Landscape recording paths.

## Dependencies

- T8 (remove dead code): CLOSED
- T9 (typed dataclasses): CLOSED — provides LLMCallRequest, LLMCallResponse, TokenUsage

## References

- Architecture analysis: `docs/arch-analysis-2026-02-22-0446/06-architect-handover.md`
- T9 contracts: `src/elspeth/contracts/call_data.py`
- Existing shared utilities: `src/elspeth/plugins/llm/__init__.py`, `validation.py`
