# Azure Document Intelligence Enrichment Transform — Design Spec

- **Date:** 2026-06-30
- **Branch:** `feature/azure-document-intelligence`
- **Status:** Design (pre-plan)
- **Plugin name:** `azure_document_intelligence`
- **Kind:** transform (row enrichment)

## 1. Problem & Goal

ELSPETH pipelines need to turn documents (PDFs, images, office files) into
structured, audit-traceable row data: extracted text/markdown, layout
(tables, paragraphs, selection marks), key-value pairs, and the typed fields
of prebuilt/custom models (invoice, receipt, ID, tax forms, …). Azure AI
Document Intelligence (formerly Form Recognizer) is the cloud service that
performs this OCR + structure extraction.

**Goal:** a transform plugin that, for each input row, submits a document
reference to Azure Document Intelligence, waits for the asynchronous analysis,
and **enriches the row** with the requested extraction facets — maximising
*compatibility* (model breadth, input modes, analyze options) and *feature
set* (every useful analyzeResult facet) while remaining **fully audited** and
**bug-free**.

This is an *enrichment* transform (adds fields, passes input through), not a
*gate* (the role of `azure_content_safety` / `azure_prompt_shield`).

## 2. Context — existing patterns we build on

| Asset | Reused how |
|-------|-----------|
| `AuditedHTTPClient` (`infrastructure/clients/http.py`) | All HTTP via this client → automatic request/response blobs, header fingerprinting (api-key never stored raw), telemetry, rate limiting. Has `.post(json=…)` and `.get()` returning raw `httpx.Response`. |
| `BatchTransformMixin` (`infrastructure/batching`) | Streaming `accept()` model: N rows processed concurrently in a worker pool, FIFO output. Same wiring as the safety transforms. |
| `BaseAzureSafetyTransform` (`transforms/azure/base.py`) | **Reference only** — NOT subclassed (see §3). We mirror its lifecycle-capture, per-`state_id` client caching, capacity-retry, and forward-invariant-probe idioms. |
| `azure/errors.py::MalformedResponseError` | Reused verbatim for Tier-3 fail-closed parsing. |
| `infrastructure/url_validation.validate_credential_safe_https_url` | Endpoint HTTPS enforcement. |
| `infrastructure/clients/json_utils.parse_json_strict` | Tier-3 strict JSON parse (rejects NaN/Infinity). |
| Enrichment idiom from `web_scrape.py` | `output = row.to_dict(); output[field] = …; narrow_contract_to_output → _apply_declared_output_field_contracts → _align_output_contract; TransformResult.success(PipelineRow(...), success_reason={"action":"enriched", …})`. |

**No new third-party dependency.** We use `httpx`/`AuditedHTTPClient`, **not**
the `azure-ai-documentintelligence` SDK — the SDK would bypass the audit trail,
and the GA REST shape we need is small and stable.

## 3. Architecture decision — self-contained, not subclassing the safety base

Three options were weighed:

- **A. Subclass `BaseAzureSafetyTransform`.** Rejected. Its
  `_process_single_with_state` is a *multi-field, fail-closed validate* loop
  that passes the row through unchanged. Document Intelligence is *one document
  reference per row* producing an *enriched* row, and the call is an
  **asynchronous long-running operation (LRO)**, not a single synchronous POST.
  We would override nearly every meaningful method and fight the base.
- **B. Extract a neutral `BaseAzureHTTPTransform` and rebase both.** Rejected
  for v1. It forces a `source_file_hash` re-roll and a full re-test of two
  hash/judge-gated security plugins (`content_safety`, `prompt_shield`) for
  **zero behaviour change** — the wrong risk for a bug-free-first goal. It is a
  clean *follow-up* once DI is proven.
- **C. Self-contained `AzureDocumentIntelligence(BaseTransform,
  BatchTransformMixin)`.** **Chosen.** ~150 lines of contained transport code
  (client caching + capacity-retry + lifecycle capture) tailored to LRO +
  enrichment. No coupling to security-gated code, focused and independently
  testable.

This is a conscious, documented trade-off: a small, bounded duplication now in
exchange for isolation and a tight blast radius. Consolidating onto a shared
transport base is recorded as future work (§11).

## 4. Configuration model

Config class: `AzureDocumentIntelligenceConfig(TransformDataConfig)` with
`model_config = {"extra": "forbid"}`. Credential fields use `repr=False`.

### 4.1 Connection
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `endpoint` | `str` (required) | — | Validated by `validate_credential_safe_https_url`. |
| `api_key` | `str` (required, `repr=False`) | — | Non-empty; sent as `Ocp-Apim-Subscription-Key`. |
| `api_version` | `str` | `"2024-11-30"` | Allowlist-validated against `_SUPPORTED_API_VERSIONS = {"2024-11-30"}` (the GA version). Configurable + trivially extensible; v1 commits to the GA `analyzeResult` shape only (§9.3). |
| `model_id` | `str` (required) | — | Any prebuilt or custom model. Mirrors Azure's server pattern `^[a-zA-Z0-9][a-zA-Z0-9._~-]{1,63}$` (maxLength 64). |

### 4.2 Document input
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `source_mode` | `Literal["url","base64"]` (required) | — | How the document reference is read from the row. |
| `source_field` | `str` (required) | — | Row field holding the URL (`url`) or base64 string (`base64`). |
| `max_base64_chars` | `int` | `8_000_000` | `base64` mode only. Bounds the audited request blob (~6 MB document). `gt=0`. |

`url` mode → request body `{"urlSource": <value>}`. `base64` mode → request
body `{"base64Source": <value>}`. Both are JSON bodies the audited client can
send; raw-binary octet-stream upload is intentionally out of scope (§11) — the
audited transport sends JSON only.

### 4.3 Output / enrichment facets
At least one output target MUST be configured (else config error).

| Field | Type | Default | Emits |
|-------|------|---------|-------|
| `content_field` | `str \| None` | `None` | `analyzeResult.content` (full text/markdown). |
| `output_content_format` | `Literal["text","markdown"]` | `"text"` | `outputContentFormat` query param. `markdown` requires a capable `api_version` (§9.3). |
| `extract` | `ExtractFields` | `ExtractFields()` | Per-facet → row-field map; only set facets are emitted. `figures` reads `analyzeResult.figures` *metadata* (regions/spans), distinct from the `output=figures` artifact-generation option (out of scope, §11). |
| `page_count_field` | `str \| None` | `None` | Number of analyzed pages (`int`) — useful for cost/routing. |
| `result_field` | `str \| None` | `None` | The **entire** `analyzeResult` object (max-fidelity escape hatch). |

`ExtractFields` (sub-model, `extra="forbid"`, each `str | None = None`):
`pages`, `tables`, `key_value_pairs`, `paragraphs`, `documents`, `languages`,
`styles`, `figures`, `sections`. Facet values are JSON-native structures
(list/dict) emitted as Python objects — verified to round-trip through contract
propagation (inferred as `object`/"any"), FIXED-mode validation, and canonical
audit hashing.

**Every declared output field is ALWAYS set** on a successful row — empty
container (`[]`/`{}`/`""`) when the model produced nothing for that facet. This
upholds `passes_through_input = True`.

All output field names (content/page_count/result + every set `extract.*`) must
be mutually distinct and are collected into `declared_output_fields` for the
executor's collision check.

### 4.4 Analyze options
| Field | Type | Default | Query param |
|-------|------|---------|-------------|
| `pages` | `str \| None` | `None` | `pages` (e.g. `"1-3,5,7-9"`); mirrors Azure's pattern `^(\d+(-\d+)?)(,\s*(\d+(-\d+)?))*$`. |
| `locale` | `str \| None` | `None` | `locale` (e.g. `"en-US"`). |
| `string_index_type` | `Literal["textElements","unicodeCodePoint","utf16CodeUnit"]` | `"textElements"` | `stringIndexType`. |
| `features` | `list[str]` | `[]` | `features` (comma-joined). Subset of the known enum {`ocrHighResolution`,`languages`,`barcodes`,`formulas`,`keyValuePairs`,`styleFont`,`queryFields`} (all native to GA). |
| `query_fields` | `list[str]` | `[]` | `queryFields` (comma-joined). Requires `"queryFields"` in `features`; non-empty ⇔ feature present. |

### 4.5 Operation control
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `poll_interval_seconds` | `float` | `1.0` | `gt=0`. Initial poll delay. |
| `poll_backoff_multiplier` | `float` | `1.5` | `ge=1`. Exponential poll backoff. |
| `poll_max_interval_seconds` | `float` | `10.0` | `gt=0`. Backoff cap. |
| `poll_timeout_seconds` | `float` | `300.0` | `gt=0`. Outer bound on the poll **sequence**. |
| `request_timeout_seconds` | `float` | `60.0` | `gt=0`. Per HTTP call (POST/GET) timeout. |
| `max_response_body_bytes` | `int` | `50_000_000` | `gt=0`. Cap on the analyzeResult body (large multi-page hi-res). |
| `max_capacity_retry_seconds` | `int` | `3600` | `gt=0`. Per-call 429/503 retry budget. |
| `batch_wait_timeout_seconds` | `int` | `3600` | `gt=0`. Batch-mixin per-row wait. |

### 4.6 Standard transform settings
`schema` (required; typically `{mode: observed}`), `on_error`, `on_success`
via `TransformDataConfig`. `declared_input_fields` initialised from config.

### 4.7 Cross-field validation (`model_validator(mode="after")`)
1. At least one output target (`content_field` or any `extract.*` or
   `result_field` or `page_count_field`).
2. All configured output field names mutually distinct.
3. `api_version ∈ _SUPPORTED_API_VERSIONS` (clear error naming the GA value).
4. Every `features` member is in the known enum.
5. `"queryFields" in features` ⇔ `query_fields` non-empty.
6. `poll_max_interval_seconds >= poll_interval_seconds`.

(Markdown and all `features` are native to the GA version, so no per-version
feature gating is required — collapsing to one GA version removes that whole
class of validation.)

## 5. Execution model

`AzureDocumentIntelligence(BaseTransform, BatchTransformMixin)`:

```
name = "azure_document_intelligence"
determinism = Determinism.EXTERNAL_CALL
plugin_version = "1.0.0"
source_file_hash = "sha256:…"        # computed post-write via scripts/cicd/plugin_hash
config_model = AzureDocumentIntelligenceConfig
passes_through_input = True
creates_tokens = False
discovery_secret_requirements = {"api_key": ("AZURE_DOCUMENT_INTELLIGENCE_KEY",)}
audit_characteristics = frozenset({AuditCharacteristic.CREDENTIALS})
capability_tags = ("azure", "document", "ocr", "enrichment", "http")
```

### 5.1 Lifecycle (mirrors `BaseAzureSafetyTransform`)
- `on_start(ctx)` — capture `recorder`, `run_id`, `telemetry_emit`, `limiter`.
- `connect_output(output, max_pending)` — `init_batch_processing(...)`.
- `accept(row, ctx)` — `accept_row(row, ctx, self._process_row)`.
- `process(...)` — raises `NotImplementedError` (use `accept`).
- per-`state_id` `AuditedHTTPClient` cache (default headers
  `{"Ocp-Apim-Subscription-Key": api_key}`; `Content-Type: application/json`
  is set per-request by `post`). Header injection is isolated in
  `_default_request_headers()` so AAD bearer auth is a clean later add.
- `close()` — set shutdown `Event`, `shutdown_batch_processing()`, close clients.

### 5.2 Per-row processing (`_process_single_with_state`)
1. **Read & validate input ref.** `source_field` present and `str`; else
   non-retryable error (`reason: "missing_field"` / `"non_string_field"`).
   `url` mode: well-formed `http(s)` URL; `base64` mode: non-empty and
   `len ≤ max_base64_chars` (else `reason: "base64_too_large"`).
2. **Submit (POST).** `{endpoint}/documentintelligence/documentModels/
   {model_id}:analyze?_overload=analyzeDocument&api-version=<v>` + analyze query
   params; JSON body (`urlSource`/`base64Source`); `Content-Type:
   application/json`. (`_overload=analyzeDocument` matches the GA reference for
   the JSON-body overload.) Wrapped in the **inner** capacity-retry loop
   (429/503 within `max_capacity_retry_seconds`).
3. **Expect 202 + `Operation-Location`.** Header absent → `MalformedResponseError`
   (Tier-3: never fabricated). Non-202 success codes → malformed. If a
   `Retry-After` header is present, it seeds the initial poll delay (clamped to
   `poll_max_interval_seconds`).
4. **Host-pin (SECURITY).** The `Operation-Location` URL host MUST equal the
   configured endpoint host. Mismatch → non-retryable error
   (`reason: "operation_location_host_mismatch"`) and we **do not** poll —
   prevents sending `Ocp-Apim-Subscription-Key` to an attacker host.
5. **Poll (GET) — outer LRO budget.** Until terminal status or
   `poll_timeout_seconds`: GET the operation URL (each GET wrapped in the inner
   capacity-retry loop), `parse_json_strict`, read `status`:
   - `notStarted` / `running` → sleep `min(interval, remaining)` (interruptible
     via shutdown `Event`), `interval *= backoff` capped at max, loop.
   - `succeeded` → take `analyzeResult`, go to step 6.
   - `failed` → non-retryable error (`reason: "analysis_failed"`); Azure
     `error.code` (a bounded enum-like token) MAY be included, but the raw
     `error.message` is **not** placed where it egresses (§7).
   - any other / missing → `MalformedResponseError` → error.
   - timeout → non-retryable error (`reason: "poll_timeout"`).
6. **Extract facets** (pure functions, `document_intelligence_result.py`) into
   the enriched output; always set every declared field.
7. **Build output contract & return.** `narrow_contract_to_output` →
   `_apply_declared_output_field_contracts` → `_align_output_contract`;
   `TransformResult.success(PipelineRow(output, contract), success_reason=…)`.
8. **finally** pop & close the cached client for this `state_id`.

### 5.3 Nested budgets & shutdown
- **Inner:** capacity-retry wraps each *individual* POST/GET (429/503), bounded
  by `max_capacity_retry_seconds`, exponential 0.05→1.0 s.
- **Outer:** the poll loop is bounded by `poll_timeout_seconds`; a 429 on a
  poll GET feeds the inner capacity loop, not the outer poll counter.
- A single `Event` (set in `close()`) interrupts both the capacity backoff and
  the poll sleep, so shutdown is prompt.

### 5.4 Replay
The poll loop is **data-driven** — it terminates on terminal `status`, never on
a wall-clock iteration count — so a recorded run and its replay observe the
identical GET sequence and terminate identically. (Poll sleeps still elapse
during replay; skipping them under replay is a documented v1.1 latency
optimisation, not a correctness issue.)

## 6. Output / enrichment contract

Mirrors `web_scrape`. `_build_output_schema_config` merges
`declared_output_fields` into `guaranteed_fields` so downstream
`required_input_fields` validate. `success_reason`:

```python
{
  "action": "enriched",
  "fields_added": sorted(declared_output_fields),
  "metadata": {                      # audit-only provenance (AGENTS.md test)
    "model_id": ..., "api_version": ..., "operation_id": ...,
    "page_count": ..., "content_format": ..., "features": [...],
    "result_status": "succeeded",
  },
}
```

No audit-only *row* fields are added, so no `*_AUDIT_FIELDS` constant is needed
(provenance lives in `success_reason`, retrievable via `elspeth explain`).

## 7. Error handling & security seams

- **Endpoint** HTTPS-validated; **api_key** non-empty, `repr=False`, never
  logged; fingerprinted (not stored raw) by `AuditedHTTPClient`.
- **Operation-Location host-pin** before polling (§5.2.4) — the linchpin
  against api-key exfiltration.
- **base64 size guard** bounds the audited request blob.
- **`urlSource`** is per-row data forwarded to Azure (Azure fetches it; we do
  not), so no SSRF pinning; we only check it is a well-formed `http(s)` URL. A
  SAS token embedded in `urlSource` is recorded in the audited request blob —
  documented in the plugin docstring.
- **Tier-3 zero-trust:** every response parsed with `parse_json_strict`,
  structure/type validated, fail-closed (`MalformedResponseError`) on anything
  unexpected — never fabricate absent fields.
- **No raw external text egress.** Per the open egress discipline
  (elspeth-30416e67cc), error results use bounded, stable `reason` codes and do
  **not** carry Azure's free-text `error.message` into `runs.error` / HTTP
  status / SSE. Network errors re-raise as `PluginRetryableError`; structural
  failures return non-retryable `TransformResult.error` with a reason dict.

**Error categories.** `TransformResult.error(reason)` requires `reason["reason"]`
to be a member of the closed `TransformErrorCategory` Literal
(`contracts/errors.py`) — a Tier-1 audit-write guard crashes on an invalid
category. The transform therefore **adds four DI-specific categories** to that
Literal (the established pattern — cf. web_scrape's `non_text_content_type`,
batch_replicate's `invalid_copies`): `analysis_failed`, `poll_timeout`,
`operation_location_missing`, `operation_location_untrusted`. All other states
map onto existing categories, carrying specificity in declared
`TransformErrorReason` keys (`error_type`, `cause`, `status_code`, `message`,
`field`, `actual_type`, `elapsed_seconds`, `max_seconds`):

| State | category | extra keys |
|-------|----------|-----------|
| source field absent | `missing_field` | `field` |
| source field non-string | `non_string_field` | `field`, `actual_type` |
| malformed `urlSource` | `invalid_input` | `field`, `error_type="invalid_document_url"` |
| base64 over size cap | `invalid_input` | `field`, `error_type="base64_too_large"`, `message` |
| submit non-202 | `api_error` | `error_type="submit_rejected"`, `status_code` |
| 202 lacks Operation-Location | `operation_location_missing` *(new)* | — |
| Operation-Location host ≠ endpoint | `operation_location_untrusted` *(new)* | — |
| poll GET non-2xx | `api_error` | `error_type="poll_request_failed"`, `status_code` |
| Azure status `failed` | `analysis_failed` *(new)* | `cause`=Azure `error.code` (bounded token; **no** raw `error.message`) |
| LRO exceeds `poll_timeout` | `poll_timeout` *(new)* | `elapsed_seconds` |
| bad JSON / unknown status / missing analyzeResult | `malformed_response` | `error_type` |
| capacity (429/503) budget exhausted | `retry_timeout` | `status_code`, `elapsed_seconds`, `max_seconds` |
| shutdown during backoff/poll | `shutdown_requested` | `elapsed_seconds` |

## 8. Audit & provenance

Every POST and every poll GET is an audited call (request/response blobs,
fingerprinted headers, latency, telemetry `ExternalCallCompleted`). The full
LRO is therefore reconstructable from the audit trail. `determinism =
EXTERNAL_CALL` records the node as non-reproducible; `audit_characteristics`
adds the `CREDENTIALS` chip.

## 9. Compatibility matrix

### 9.1 Models (`model_id`, any string passes format check)
`prebuilt-read`, `prebuilt-layout`, `prebuilt-document`, `prebuilt-invoice`,
`prebuilt-receipt`, `prebuilt-idDocument`, `prebuilt-tax.us.w2` (and other
`prebuilt-tax.*`), `prebuilt-healthInsuranceCard.us`, `prebuilt-contract`,
`prebuilt-marriageCertificate.us`, `prebuilt-creditCard`, custom model IDs.
Document **classifiers** (`documentClassifiers/{id}:classify`) use a different
endpoint and are out of scope for v1 (§11).

### 9.2 Input modes
`urlSource` (incl. SAS-tokened blob URLs) and `base64Source` (size-guarded).

### 9.3 API version
v1 supports **`2024-11-30` (GA) only**, enforced by an allowlist at config time
(`_SUPPORTED_API_VERSIONS`). The default IS the GA version; an operator who sets
anything else gets a clear error naming the supported value. Markdown,
`queryFields`, `figures`, and all add-on `features` are native to GA, so a
single version covers the full feature set with no cross-version parse matrix —
the deliberate "compatibility = breadth on one stable GA contract" trade-off.
Adding a future GA version is a one-line allowlist addition plus a parser
review; the strict Tier-3 parser already fail-closes on any unexpected shape.

### 9.4 Auth
`api_key` for v1 (matches the safety transforms exactly). Header injection is
isolated so AAD / `DefaultAzureCredential` bearer auth is a clean later add
(§11) — token-refresh-under-concurrency is deliberately not built now.

## 10. Testing strategy

- **Unit** (`tests/unit/plugins/transforms/azure/test_document_intelligence.py`):
  config validation (every rule in §4.7), LRO happy path via a fake client
  (POST→202+`Operation-Location`, GET `running` then `succeeded`), facet
  extraction for each declared facet (and empty-container defaults), markdown +
  feature/version gating, and every error path in §7 (missing/non-string field,
  base64 oversize, missing/mismatched `Operation-Location`, `failed` status,
  `poll_timeout`, malformed result). Assert error reasons carry no raw Azure
  `error.message`.
- **Facet parser** unit tests on the pure functions in
  `document_intelligence_result.py` (no I/O).
- **Contract invariant**
  (`tests/unit/contracts/transform_contracts/test_azure_document_intelligence_contract.py`):
  forward/backward ADR-009 probes, mirroring the safety contract tests, using
  `probe_config()` + a local fake client in `execute_forward_invariant_probe`.
- **Golden knob schema**
  (`tests/golden/web/catalog/knob_schema/transform__azure_document_intelligence.json`):
  generated to match the catalog build.
- **Property** (optional, `tests/property/plugins/transforms/azure/`): config
  round-trips and facet-extraction invariants over Hypothesis-generated
  analyzeResult shapes.
- Full local lint set (ruff, mypy) + `pytest tests/` slice + `wardline scan`.

## 11. Out of scope / future

- Raw octet-stream binary upload (needs audited-client binary-body support).
- `output=pdf` / `output=figures` generated-artifact retrieval (searchable PDF,
  cropped figure images via the separate `analyzeResults/{id}/pdf|figures`
  GETs). The `analyzeResult.figures` *metadata* facet is supported.
- Multi-GA-version support (only `2024-11-30` in v1).
- Document **classification** (`:classifyDocument`).
- AAD / `DefaultAzureCredential` bearer auth (header seam is ready).
- A shared `BaseAzureHTTPTransform` consolidating safety + DI transport
  (Option B) — a clean refactor once DI is proven.
- Replay-aware poll-sleep skipping.

## 12. File-level plan

| File | Purpose |
|------|---------|
| `src/elspeth/plugins/transforms/azure/document_intelligence.py` | Config models (`ExtractFields`, `AzureDocumentIntelligenceConfig`) + `AzureDocumentIntelligence` transform (lifecycle, LRO transport, output). |
| `src/elspeth/plugins/transforms/azure/document_intelligence_result.py` | Pure Tier-3 facet extraction/validation functions (no I/O). |
| `tests/unit/plugins/transforms/azure/test_document_intelligence.py` | Unit tests (config, LRO, facets, errors). |
| `tests/unit/contracts/transform_contracts/test_azure_document_intelligence_contract.py` | ADR-009 forward/backward invariant. |
| `tests/golden/web/catalog/knob_schema/transform__azure_document_intelligence.json` | Golden knob schema. |
| (optional) `tests/property/plugins/transforms/azure/test_document_intelligence_properties.py` | Property tests. |

No `pyproject.toml` change (no new dependency). `source_file_hash` computed and
set after the source file is written.
