# L2 #4 — `plugins/` cluster catalog

## Conventions

- One entry per immediate sub-subsystem (4 entries: `infrastructure/`, `sources/`, `transforms/`, `sinks/`) per Δ L2-3.
- `infrastructure/` and `transforms/` qualify as composite at L2 depth (Δ4 fires); both are flagged "L3 candidate: composite at L2 depth" without third-level recursion. Their entries describe responsibility, file/sub-pkg structure, intra-cluster edges, patterns observed at the entry-point/docstring layer, and concerns — but do **not** open file bodies past the first ~30 lines.
- Catalog is presented in **F3 reading order**: `infrastructure/` first (the spine), then `sinks/`, `sources/`, `transforms/` as clients.
- Citations: file paths are project-root-relative within `src/elspeth/plugins/`; oracle citations use `[ORACLE: …]`; KNOW citations use `[CITES KNOW-…]` or `[DIVERGES FROM KNOW-…]`.
- Cross-cluster boundary: per Δ L2-4, claims about other clusters' internal structure are forbidden; only edges from `temp/intra-cluster-edges.json` and the L3 oracle are cited.
- SCC handling (Δ L2-7): the `transforms/` entry marks members of SCC #1 explicitly.

## Entry 1 — `plugins/infrastructure/`

**Path:** `src/elspeth/plugins/infrastructure/`

**Responsibility:** The plugin ecosystem's spine — pluggy hookspecs, the plugin manager (`PluginManager`), dynamic discovery (folder-scan with `issubclass`-driven detection), the base classes (`BaseSource`, `BaseTransform`, `BaseSink`), Pydantic-based plugin config bases, audited HTTP/LLM clients (the audit-trail wiring point for KNOW-C9 / KNOW-C10), batching ports for plugin-internal pipelining, and pooling infrastructure for parallel API calls with AIMD throttling. **L3 candidate: composite at L2 depth** (Δ4 fires — 41 files, 10,782 LOC, 3 sub-packages plus `clients/retrieval/`).

**File count, LOC:** 41 files; 10,782 LOC. Sub-package breakdown: root 16 files / 3,804 LOC; `clients/` 9 files / 3,790 LOC + nested `clients/retrieval/` 6 files / 1,031 LOC; `batching/` 4 files / 1,024 LOC; `pooling/` 6 files / 1,133 LOC.

**Internal coupling** (cites `temp/intra-cluster-edges.json`): inbound from siblings — `plugins/sinks` (w=45), `plugins/transforms` (w=40), `plugins/sources` (w=17), `plugins/transforms/llm` (w=17), `plugins/transforms/azure` (w=6), `plugins/transforms/llm/providers` (w=12 to `clients/`), `plugins/transforms/rag` (w=9 to `clients/retrieval/`), `plugins/sources` (w=5 to `clients/`). **No outbound edges to siblings** — `infrastructure/` is the spine.

**External coupling** (cites L3 oracle): inbound from other clusters — `web/composer` (w=22), `.` cli root (w=7), `web/execution` (w=4), `testing` (w=4), `web/catalog` (w=3), `web` (w=1). 0 outbound L3↔L3 edges; downward-only (contracts/core/engine, not in this graph).

**Patterns observed:**

- **Pluggy hookspecs at `infrastructure/hookspecs.py:1-25`** — `PROJECT_NAME`, `ElspethSourceSpec`, `ElspethSinkSpec`, `ElspethTransformSpec` define the registration interface. Module docstring at lines 1–18 is explicit: "@hookspec defines the hook interface (done here). @hookimpl marks plugin implementations of those hooks." This is the architectural anchor for [CITES KNOW-C22] "ELSPETH uses pluggy for clean architecture, NOT to accept arbitrary user plugins."
- **Discovery is `issubclass`-based, not `Protocol`-based** (`infrastructure/base.py:7-15` docstring): "Plugin discovery uses issubclass() checks against base classes / Python's Protocol with non-method members (name, determinism, etc.) cannot support issubclass()." Aligns with [CITES KNOW-C25] (no `getattr(x, 'attr', default)`) — the framework chose subclassing precisely so missing attributes crash via Python's normal MRO, not via defensive lookups.
- **Dynamic plugin discovery via folder scan** (`infrastructure/discovery.py:1-30`): scans plugin directories for non-abstract subclasses with a `name` class attribute. `EXCLUDED_FILES` (`discovery.py:18-26`) hard-codes `__init__.py`, `hookimpl.py`, `base.py`, `templates.py`, `auth.py`, `sentinels.py` — files that contain helpers rather than registrable plugins.
- **Lifecycle contract is explicit** (`infrastructure/base.py:21-35`): `on_start(ctx) -> [process/load/write] -> on_complete(ctx) -> close()`. `on_start` failure short-circuits; `on_complete` runs even on pipeline crash; `close` is "pure resource teardown (no context)." This is the engine-side ABC for the plugin runtime contract.
- **Audited clients are the audit-trail wiring point** (`clients/__init__.py:1-15`): "These clients wrap external service calls (LLM, HTTP) and ensure every request/response is recorded to the Landscape audit trail for complete traceability." Direct evidence for [CITES KNOW-C9] and [CITES KNOW-C10] (attributability test) at the plugin layer.
- **SSRF defence at the HTTP boundary** (`clients/http.py:3-5`): "Provides SSRF-safe HTTP methods via get_ssrf_safe() which uses IP pinning to prevent DNS rebinding attacks. See core/security/web.py for details." — security control wired below the audit envelope.
- **Replay mode for deterministic re-execution** (`clients/replayer.py:1-11`): `CallReplayer` matches by request_hash; in replay mode external calls return recorded responses. Direct support for [CITES KNOW-C7] (auditability) and [CITES KNOW-C10] (re-execution from audit trail).
- **Plugin-internal concurrency is hidden behind ports** (`batching/__init__.py:1-15`, `batching/ports.py:1-15`): every pipeline stage has input/output ports; "the orchestrator sees synchronous behavior; concurrency is hidden inside the plugin boundary." This is how transforms can pipeline without leaking concurrency upward.
- **Pooling with AIMD throttling** (`pooling/executor.py:1-12`): `PooledExecutor` manages concurrent API calls under semaphore limits with adaptive throttle delays, per-state_id caching, reorder-buffered output. The `pooling/__init__.py:1-17` re-exports `AIMDThrottle`, `BufferEntry`, `CapacityError`, `PoolConfig`, `PooledExecutor`, `RowContext`, `ThrottleConfig`, `is_capacity_error`.

**Test evidence:**

- `tests/unit/plugins/test_discovery.py` — discovery contract (asserts that file-scan finds subclasses, respects `EXCLUDED_FILES`, etc.).
- `tests/unit/plugins/test_manager.py`, `test_manager_singleton.py`, `test_hookimpl_registration.py` — pluggy registration paths.
- `tests/unit/plugins/test_base.py`, `test_base_signatures.py`, `test_base_sink_contract.py`, `test_base_source_contract.py` — base-class invariants.
- `tests/unit/plugins/infrastructure/test_base_semantics.py`, `test_probe_factory.py` — invariant probes.
- `tests/unit/plugins/clients/` — audited-client unit tests.

**Concerns:**

- **R-rule findings concentration in `clients/` and `pooling/`** is expected (this is the L3 boundary that wraps external SDKs). Whether each instance is allowlist-justified is allowlist-mediated and not in scope for this pass; however, the catalog cannot assert "no problematic defensive patterns" at L2 depth without opening the bodies of the 16 files in `infrastructure/` root + 9 in `clients/`. This is a **debt candidate** (test-coverage of allowlist coherence).
- **`infrastructure/__init__.py` is essentially empty** (1 LOC: just the docstring). All public surface is via direct module imports from siblings. There is no curated `__all__` list at the package root — discoverability of the spine is documentation-driven, not import-driven. **Debt candidate** (a stable contract surface for the spine would simplify F3 reading-order onboarding).
- **`base.py` (1,159 LOC)** sits below the L3-deep-dive threshold (1,500) but represents ~30% of the root-level `infrastructure/` LOC concentrated in a single file containing all three base classes. Whether splitting `BaseSource`, `BaseTransform`, `BaseSink` into separate modules would help is an architecture-pack question; this pass surfaces only.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §4 plugins/` (sub-area enumeration "infrastructure/ (hookspecs, audited clients, base classes)") and `04-l1-summary.md §F3` (oracle citation for the spine role). [CITES KNOW-A16] [CITES KNOW-C9] [CITES KNOW-C10] [CITES KNOW-C21] [CITES KNOW-C22] [CITES KNOW-C25] [CITES KNOW-P3] [CITES KNOW-P4] [CITES KNOW-P5]

**Confidence:** **High** — sub-package structure and LOC verified by `find` + `wc -l`; pattern claims cite first-30-lines of entry-points; cross-cluster edges cite the L3 oracle directly; tests enumerated by `ls`. **L3-deep-dive deferral is mandatory** (composite at L2 depth) — bodies of `base.py` (1,159 LOC), `clients/http.py` (854 LOC), `clients/llm.py` (719 LOC), `pooling/executor.py` (651 LOC), and `config_base.py` (544 LOC) are not opened.

---

## Entry 2 — `plugins/sinks/`

**Path:** `src/elspeth/plugins/sinks/`

**Responsibility:** Output destinations for pipeline rows — write rows to CSV / JSON / Azure Blob / database / Dataverse / Chroma vector store. Multiple sinks per run (KNOW-C26 ACT). Coercion-forbidden (`allow_coercion=False`); wrong types are upstream bugs, expected to crash.

**File count, LOC:** 7 files (flat, no sub-packages); 3,515 LOC. Plugin classes registered: 6 (`csv`, `json`, `azure_blob`, `dataverse`, `database`, `chroma_sink`).

**Internal coupling** (cites `temp/intra-cluster-edges.json`): outbound to `plugins/infrastructure` (w=45) — **the heaviest single L3 edge in the codebase**. No other intra-cluster edges (sinks do not import from sources or transforms).

**External coupling:** No L3↔L3 inbound edges from other clusters direct to `plugins/sinks/` at the package level — sinks are instantiated through `plugins/infrastructure`'s `PluginManager` factory by callers (engine, web/composer via the manager).

**Patterns observed:**

- **Trust-tier docstring is universal** — every sink module repeats the contract verbatim:
  - `sinks/json_sink.py:5-7`, `csv_sink.py:5-7`, `database_sink.py:5-7`: "Sinks use allow_coercion=False to enforce that transforms output correct types. Wrong types = upstream bug = crash."
  - `sinks/azure_blob_sink.py:5-12` extends the contract with the explicit Tier-3-vs-OUR-CODE breakdown: "Azure Blob SDK calls = EXTERNAL SYSTEM -> wrap with try/except / Serialization of rows = OUR CODE -> let it crash / Internal state = OUR CODE -> let it crash." This is verbatim alignment with [CITES KNOW-C13] [CITES KNOW-C14] [CITES KNOW-C19] [CITES KNOW-C20].
- **Content hashing for audit integrity** (`csv_sink.py:1-3`, `database_sink.py:1-7`): each sink hashes its written content and records the hash to the audit trail (KNOW-C7 / KNOW-C8 — hashes survive payload deletion).
- **Sink registration is type-scoped** (`__init__.py:1-9`): "Plugins are accessed via PluginManager, not direct imports." [CITES KNOW-C21] enforced at import boundary.
- **Pydantic-validated config** — every sink's config inherits from `infrastructure/config_base.py`'s base classes (e.g., `PathConfig`, sink-specific subclasses). Config validation rejects unknown fields (strict).

**Test evidence:**

- `tests/unit/plugins/sinks/` (subdirectory exists; per-sink unit tests).
- `tests/unit/plugins/test_base_sink_contract.py` — base contract test.
- `tests/integration/plugins/sinks/` — integration tests (sink × real backend).

**Concerns:**

- **R-rule findings concentrate here** — from the `temp/layer-check-plugins.txt` (defensive-pattern scanner): `sinks/azure_blob_sink.py:198` (R5 isinstance on `headers`), `sinks/chroma_sink.py:273` (R6 except KeyError), `sinks/chroma_sink.py:281,288` (R5 isinstance on `raw_id`/`raw_doc`), and others. Most are at the Tier-3 boundary handling external SDK responses (Azure Blob, Chroma) — likely allowlist-justified, but the catalog cannot verify at L2 depth without opening the bodies.
- **`sinks/__init__.py` exposes no `__all__`**; the package re-exports nothing. Sinks are accessed only by registered name through `PluginManager`. This is intentional (system-ownership) but means the import surface is invisible without reading the manager.

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §4 plugins/` ("sinks/" sub-area). [CITES KNOW-A35 (count)] [DIVERGES FROM KNOW-A35 (count drift)] [CITES KNOW-C13] [CITES KNOW-C14] [CITES KNOW-C19] [CITES KNOW-C20] [CITES KNOW-C21] [CITES KNOW-C22] [CITES KNOW-P5].

**Confidence:** **High** — flat layout, 7 files all read at entry-point depth; trust-tier discipline verifiable by docstring matching; oracle edge `sinks → infrastructure (w=45)` is the unique reference point.

---

## Entry 3 — `plugins/sources/`

**Path:** `src/elspeth/plugins/sources/`

**Responsibility:** Pipeline data ingestion — load rows from CSV / JSON / Azure Blob / Dataverse / text / null. Exactly one source per run (KNOW-C26 SENSE). Coercion-allowed (`allow_coercion=True`) — **the only place in the pipeline where coercion is permitted**, per [CITES KNOW-C19].

**File count, LOC:** 8 files (flat, no sub-packages); 3,519 LOC. Plugin classes registered: 6 (`csv`, `json`, `azure_blob`, `dataverse`, `text`, `null`). Plus `field_normalization.py` (271 LOC), the shared header-normalization module.

**Internal coupling** (cites `temp/intra-cluster-edges.json`): outbound to `plugins/infrastructure` (w=17) and to `plugins/infrastructure/clients` (w=5; Dataverse uses audited HTTP client). No other intra-cluster edges.

**External coupling:** Inbound from `.` cli root (w=2) — likely the `TRANSFORM_PLUGINS`-style registry at `cli.py` (per KNOW-P22 noted in the L1 standing note). No other L3↔L3 inbound edges direct to `plugins/sources/`.

**Patterns observed:**

- **Coercion-allowed contract is universal** — every source module repeats verbatim:
  - `sources/csv_source.py:5-6`, `json_source.py:5-6`, `azure_blob_source.py:5-6`, `dataverse.py:6-7`, `text_source.py:6-7`: "Sources use allow_coercion=True to normalize external data. This is the ONLY place in the pipeline where coercion is allowed." Direct alignment with [CITES KNOW-C18] [CITES KNOW-C19].
- **Field-name normalisation at the boundary** (`field_normalization.py:1-15`): "normalizes messy external headers (e.g., 'CaSE Study1 !!!! xx!') to valid Python identifiers (e.g., 'case_study1_xx') at the source boundary." The algorithm is **versioned** (`NORMALIZATION_ALGORITHM_VERSION`) and stored in the audit trail "to enable debugging cross-run field name drift when algorithm evolves." This is the canonical Tier-3 normalisation surface — also a clear [CITES KNOW-C18] (sources MAY coerce; coercion is meaning-preserving).
- **Quarantine behaviour** is implied across sources but not directly visible at entry-point depth — bodies of `csv_source.py` (515 LOC), `json_source.py` (603 LOC), etc. handle malformed-row quarantine. This is the load-bearing Tier-3 contract; not opened here per L2 depth cap.
- **NaN / Infinity rejection** (`json_source.py:7-8`): "Non-standard JSON constants (NaN, Infinity, -Infinity) are rejected at parse time per canonical JSON policy. Use null for missing values." Aligns with the "absence is not fabrication" rule — null preserved, NaN refused.
- **`NullSource` is special-purpose** (`null_source.py:1-5`): "Used by resume operations where row data comes from the payload store, not from the original source. Satisfies PipelineConfig.source typing while actual row data is retrieved separately." This is how resume bypasses the source-yields-rows contract while keeping the type system happy.

**Test evidence:**

- `tests/unit/plugins/sources/` — per-source unit tests.
- `tests/unit/plugins/test_base_source_contract.py` — base contract test.
- `tests/property/plugins/test_schema_coercion_properties.py` — Hypothesis-based property tests for coercion behaviour at the source boundary (Tier 3).
- `tests/integration/plugins/sources/` — integration tests (source × real backend).

**Concerns:**

- **`field_normalization.py` is the single largest source-side module** (271 LOC) and encodes a versioned algorithm. The audit-trail-stored version (`NORMALIZATION_ALGORITHM_VERSION`) needs an explicit per-version test that locks in the algorithm output for known input strings — without one, version bumps are silent. **Debt candidate** (regression-locked corpus test for the normaliser).
- **No source-side coercion-allowed test that asserts it's the *only* place** — the discipline is verbal ("ONLY place") and structural (allow_coercion flag), but there's no integration test that fails if a transform or sink is observed to coerce. **Debt candidate** (cross-cluster invariant test).

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §4 plugins/` ("sources/" sub-area). [CITES KNOW-C18] [CITES KNOW-C19] [CITES KNOW-C26] [CITES KNOW-P3] [CITES KNOW-P6].

**Confidence:** **High** — flat layout, all 8 files read at entry-point depth; trust-tier docstrings verifiable; oracle edges cite directly.

---

## Entry 4 — `plugins/transforms/`

**Path:** `src/elspeth/plugins/transforms/`

**Responsibility:** Per-row processing in the DECIDE stage of SDA — row transforms, batch-aware transforms, LLM transforms (single + multi-query + batch), Azure safety transforms (content safety, prompt shield), RAG retrieval, web scraping, value/field manipulation, type coercion guards. **L3 candidate: composite at L2 depth** (Δ4 fires — 41 files, 12,575 LOC, 3 sub-packages `azure/`, `llm/`, `rag/` plus nested `llm/providers/`).

**File count, LOC:** 41 files; 12,575 LOC. Plugin classes registered: 17 (see discovery findings §Plugin count).

**Internal coupling** (cites `temp/intra-cluster-edges.json`):

- Outbound to `plugins/infrastructure` (w=40) — second-heaviest L3 edge.
- Outbound to `plugins/infrastructure` from `transforms/llm` (w=17), `transforms/azure` (w=6).
- Outbound to `plugins/infrastructure/clients` from `transforms/llm/providers` (w=12).
- Outbound to `plugins/infrastructure/clients/retrieval` from `transforms/rag` (w=9).
- **SCC #1 internal:** `plugins/transforms/llm` ↔ `plugins/transforms/llm/providers` (10 + 5 = 15 weight across two edges; plus the reverse `providers → llm` recovery edges in the SCC closure).

**External coupling:** No L3↔L3 inbound edges direct to `plugins/transforms/` at the package level (transforms are instantiated through `PluginManager`).

**Patterns observed:**

- **Coercion-forbidden contract is universal** — `value_transform.py:5-7`, `passthrough.py:5-7` repeat: "Transforms use allow_coercion=False to catch upstream bugs. If the source outputs wrong types, the transform crashes immediately." Direct alignment with [CITES KNOW-C13] [CITES KNOW-C14] [CITES KNOW-C20].
- **`transforms/__init__.py` re-exports only `TypeCoerce` and `ValueTransform`** (`__init__.py:11-19`). Other transforms are accessed exclusively through `PluginManager` — the contract is "name-based registry, not import-based access" [CITES KNOW-C21].
- **Strategy pattern at the LLM transform** (`llm/transform.py:1-13`): "LLMTransform dispatches to SingleQueryStrategy or MultiQueryStrategy based on whether queries are configured. Provider dispatch (Azure, OpenRouter) is handled via _PROVIDERS registry." Uses `BatchTransformMixin` from `infrastructure/batching/`.
- **Provider protocol with deferred instantiation** (`llm/provider.py:1-15`, `llm/transform.py:9-13`): "The LLMProvider protocol defines the narrow interface between LLMTransform (shared logic) and provider-specific transport." Providers own client lifecycle, Tier-3 boundary validation, error classification, audit recording, finish-reason normalisation. **Provider instantiation is deferred to `on_start()`** — the import-time coupling that creates SCC #1 is divorced from the runtime instantiation moment.
- **Tier-3 boundary discipline at LLM providers**:
  - `llm/providers/openrouter.py:1-13`: "Handles raw HTTP transport with full Tier 3 boundary validation: JSON parsing with NaN/Infinity rejection / Content extraction from choices[0].message.content / Null content → ContentPolicyError / Non-finite usage values → LLMClientError / HTTP status code → typed exception mapping."
  - `llm/providers/azure.py:1-12`: "Thin wrapper over AuditedLLMClient that normalizes LLMResponse into LLMQueryResult."
  - Both providers cache clients **per-state_id with a threading lock**. `azure.py:8-11` adds: "The state_id is snapshot at method entry (not read from a mutable context) to prevent evicting the wrong cache entry during retry races." This is concurrency-correct lifecycle handling.
- **Azure safety transforms share a base** (`azure/base.py:1-13`): batch processing lifecycle, audited HTTP client management with per-state_id caching, rate limiting integration, recorder/telemetry capture from `LifecycleContext`, field-scanning loop. `BaseAzureSafetyTransform` is itself excluded from plugin discovery (lives in `azure/base.py`; discovery's `EXCLUDED_FILES` excludes `base.py`).
- **RAG retrieval lifecycle is explicit** (`rag/transform.py:1-12`): `__init__` → `on_start` (provider construction via PROVIDERS factory) → `process` (build query → search → format → attach) → `on_complete` (telemetry) → `close`.
- **Web scraping has explicit security controls** (`web_scrape.py:5-19`): SSRF prevention (private-IP and cloud-metadata blocks), URL scheme allowlist (HTTP/HTTPS only), configurable timeouts, rate limiting; audit trail integration via `AuditedHTTPClient` and `PayloadStore`.
- **Batch-aware transforms are explicit** (`batch_replicate.py:7-13`, `batch_stats.py:6-9`): "uses is_batch_aware=True, meaning the engine will buffer rows and call process() with a list when the trigger fires." `batch_replicate.py:11-12` adds: "For output_mode: transform, the engine creates NEW tokens for each output row, with parent linkage to track deaggregation lineage" — direct alignment with the engine's token-identity invariant.

**SCC #1 — member coupling notation (per Δ L2-7):**

- `plugins/transforms/llm` is a **member of SCC #1** with `plugins/transforms/llm/providers`. Acyclic decomposition not possible at L3↔L3 layer.
- `plugins/transforms/llm/providers` is a **member of SCC #1** with `plugins/transforms/llm`. Acyclic decomposition not possible at L3↔L3 layer.
- Specific cycle import sites already enumerated in `01-cluster-discovery.md §SCC #1 evidence` (file:line citations to `transform.py:64-65` forward and `providers/{azure,openrouter}.py:23-25/35-37` reverse).

**Test evidence:**

- `tests/unit/plugins/transforms/`, `tests/unit/plugins/llm/`, `tests/unit/plugins/llm/test_provider_protocol.py`, `test_provider_azure.py`, `test_provider_openrouter.py`, `test_provider_lifecycle.py`, `test_plugin_registration.py` — protocol, providers, registration.
- `tests/unit/plugins/llm/test_azure_batch.py`, `test_openrouter_batch.py` — batch-mode tests.
- `tests/unit/plugins/llm/test_pooled_executor.py`, `test_aimd_throttle.py` — pooling concurrency invariants.
- `tests/integration/plugins/transforms/`, `tests/integration/plugins/llm/` — end-to-end transform tests.

**Concerns:**

- **L3 deep-dive deferral:** `llm/azure_batch.py` (1,592 LOC) is over the 1,500-LOC threshold and is **flagged**, not opened. `llm/transform.py` (1,446 LOC) is under threshold but is the locus of strategy + provider dispatch; reading it would inform the SCC analysis but is deferred.
- **SCC #1 surface contract coverage:** the providers' Tier-3 validation responsibilities (NaN rejection, finish-reason normalisation, error classification) are documented in `llm/provider.py:1-15`. Whether each documented responsibility has a per-provider negative test is the kind of question that would need body-level reading.
- **R-rule findings concentrate in transforms** as well — `transforms/llm/transform.py`, `transforms/web_scrape.py`, `transforms/batch_*.py` have R5/R6 occurrences from the defensive-pattern scanner. Most are likely allowlist-justified at provider boundaries; same disclaimer as for sinks/.
- **`azure/base.py:BaseAzureSafetyTransform`** sits in a discovery-excluded file but inherits from `BaseTransform`. If the file naming convention ever changes (e.g., `azure/safety_base.py`), discovery would attempt to register the abstract base class. **Debt candidate** (an integration test that asserts `BaseAzureSafetyTransform` is *not* in the registered plugin set, regardless of filename).

**L1 cross-reference:** Supplements `02-l1-subsystem-map.md §4 plugins/` ("transforms/" sub-area, including the highest-risk concern about `azure_batch.py` 1,592 LOC). [CITES KNOW-A35] [DIVERGES FROM KNOW-A35] [CITES KNOW-C13] [CITES KNOW-C14] [CITES KNOW-C20] [CITES KNOW-C21] [CITES KNOW-C26] [CITES KNOW-C30] [CITES KNOW-P4] [CITES KNOW-P6] [CITES KNOW-P7].

**Confidence:** **High** for cluster-level claims (LOC, file counts, sub-package layout, intra-cluster edges, SCC #1 import sites, F3 reading-order verification). **Medium** for individual plugin claims that depend on body-level reading deferred per L2 depth cap (e.g., per-provider Tier-3 validation completeness).

---

## Closing

### Sub-subsystem inventory check

4 entries; 4 immediate sub-directories under `plugins/`; 1:1. No invented entries; no omissions.

### Composite-at-L2 deferrals

- `plugins/infrastructure/` — composite per Δ4 (41 files, 10,782 LOC, 3 sub-packages + nested `clients/retrieval/`).
- `plugins/transforms/` — composite per Δ4 (41 files, 12,575 LOC, 3 sub-packages + nested `llm/providers/`).

### L3 deep-dive candidates flagged (>=1,500 LOC)

| File | LOC | Reason |
|---|---:|---|
| `plugins/transforms/llm/azure_batch.py` | 1,592 | Body not opened; deferred per L2 depth cap. Likely the locus of Azure async batch API + checkpointing logic. |

### SCC handling per Δ L2-7

SCC #1 (`plugins/transforms/llm` ↔ `plugins/transforms/llm/providers`) is internal to this cluster; both members marked in the `transforms/` entry. Detailed cycle-intent surfacing in `04-cluster-report.md §SCC analysis`.

### Test-debt candidates surfaced (per Δ L2-5)

1. **Allowlist-coherence test for R-rule findings in `clients/` and `pooling/`** — currently allowlist entries decay implicitly; an invariant test that asserts every R-rule finding in `infrastructure/clients/` has a current allowlist entry would lock in the boundary discipline.
2. **Stable contract surface for `infrastructure/`** — `infrastructure/__init__.py` exposes no `__all__`; the spine has no documented import surface.
3. **Per-version corpus regression test for `field_normalization.NORMALIZATION_ALGORITHM_VERSION`** — version bumps are currently silent.
4. **Cross-cluster invariant test for "coercion only at sources"** — the discipline is documented in every plugin file but not test-locked.
5. **Negative test for `BaseAzureSafetyTransform` not appearing in plugin registry** — discovery excludes by filename, not by abstract-class status.
