# ELSPETH Progress Report — RC1 to RC5

**Period covered:** 12 January 2026 → 19 May 2026 (128 days)
**Repository state:** 4,521 unique commits across the RC-1, RC-2, and RC-5.2 history snapshots as of 19 May 2026
**Author of record:** John Morrissey
**Audience:** Engineering team and engineering leadership
**Register:** Technical
**Purpose:** Cumulative view of what ELSPETH delivered, when, and how the surface evolved from an empty scaffold to a high-assurance, web-authored, auditable pipeline platform.

This document is the **progress / outputs** view. For per-day work volume, see [elspeth-velocity-rc1-to-rc5.md](elspeth-velocity-rc1-to-rc5.md).

## How to read this report

1. **Framing.** This is a cumulative *outputs* view, not a feature roadmap and not a release record. The authoritative line-by-line release records are the archived `CHANGELOG-RC1.md` and `CHANGELOG-RC2.md` (now under `docs-archive/2026-05-19-docs-cleanout/`) plus the active `/CHANGELOG.md` (RC-3 onwards). This report aggregates and narrates those records by Period.
2. **Periods, not releases.** The report's grouping unit is the "Period" — a chunk of work bounded by phase change, not always by RC tag. Period boundaries are documented in each Period's header. Where a Period spans multiple RC tags, the RC tags appear as sub-headings inside the Period.
3. **Commit counts.** Counts are deduplicated across the RC-1, RC-2, and RC-5.2 history snapshots. Small gaps in the cumulative-commit column of the executive timeline reflect that dedup, not missing work.
4. **Date qualification.** Where the `CHANGELOG.md` does not stamp explicit release commits (notably RC-4.0 and RC-4.1), date ranges are inferred from the dominant feature-commit cluster. Inferred dates carry an in-period date callout.
5. **Per-Period *Sized* blocks.** Every Period closes with a standardised `Sized` block (commits, calendar days, active days, peak day, tests at end of period, notable infrastructure). The standardised shape is intended to make Periods diff-comparable across the project's life.
6. **What this report does not cover.** Plugin internals (see the plugin README under each plugin directory), frontend internals (see `src/elspeth/web/frontend/`), pipeline-author UX (see `pipeline-composer` skill), and operational procedures (see `docs/runbooks/`).

---

## Executive Timeline

| Release | Date | Theme | Cumulative commits | Details |
|---------|------|-------|--------------------:|---------|
| Project initiation | 2026-01-12 | First scaffold commit `748666333` | 0 | [Period 1](#period-1) |
| **RC-1** | 2026-01-22 | Auditable SDA framework ready for use | 782 | [Period 1](#period-1) |
| **RC-2 (0.1.0)** | 2026-02-02 | Telemetry, ChaosLLM, bug burndown | 1,593 | [Period 2](#period-2) |
| RC-2.1 → 2.5 | 2026-02-03 → 02-12 | Key Vault, schema contracts, `PipelineRow`, WebScrape, SQLCipher, declarative DAG | 2,120 | [Period 3](#period-3) |
| **RC-3.2 (0.3.0)** tag | 2026-02-22 | Strict typing at audit boundaries, schema contracts public | 2,294 | [Period 4](#period-4) |
| **RC-3.3 (0.3.3)** | 2026-03-02 | Architectural remediation (T10, T17–T19), repository pattern | 2,441 | [Period 4](#period-4) |
| **RC-3.4 (0.3.4)** | 2026-03-10 | Systematic hardening — 191-bug triage, deep immutability | 2,595 | [Period 4](#period-4) |
| **RC-4.0 (0.4.0)** | ~2026-03-22 → 03-29 | Dataverse + RAG plugins, output schema contracts | 2,921 | [Period 5](#period-5) |
| **RC-4.1 (0.4.1)** | ~2026-03-29 → 04-02 | ChromaSink, `depends_on`, commencement gates, readiness | 3,013 | [Period 5](#period-5) |
| **RC-5 (0.5.0)** cut | 2026-04-03 | Web UX Composer, auth, blob store, secret refs, MCP composer | 3,073 | [Period 5](#period-5) |
| **RC-5.1 (0.5.1)** | 2026-05-11 | Composer correctness + audit-integrity coverage | 3,883 | [Period 6](#period-6) |
| **RC-5.2 (0.5.2)** | 2026-05-14 | Guided composer + durable progress + recovery UX | 4,210 | [Period 7](#period-7) |
| RC-5.2 documentation snapshot | 2026-05-19 | Composer Phase 6 completion gestures, Phase 7 catalog reshape, Phase 8 polish | **4,521** | [Period 7](#period-7) |

> Commit counts are cumulative across the RC-1, RC-2, and RC-5.2 history snapshots; the small gaps reflect merge-commit dedup. Cutover dates are taken from the canonical RC commit message (`ELSPETH - Release Candidate N`), the version-bump commit, or the release tag.
>
> **Date qualification — RC-4.0 and RC-4.1:** `CHANGELOG.md` does not stamp explicit release commits for `[0.4.0]` or `[0.4.1]`. The date ranges shown are inferred from the dominant feature-commit clusters between RC-3.4 (10 Mar) and the RC-5 cut (3 Apr): Dataverse + RAG + output-schema-contracts landed around 22–29 March; ChromaSink + `depends_on` + commencement gates landed 30 March – 2 April. Read the boundary as the dominant work-cluster, not a release stamp.

---

<a id="period-1"></a>
## Period 1 — Foundation (Pre-RC1, 12–22 January 2026)

**Outcome:** Over eleven days, ELSPETH went from an empty scaffold to a working auditable Sense/Decide/Act pipeline framework, with CLI, plugin system, full DAG executor, LLM integration, Azure ecosystem support, and crash-safe recovery.

### Capabilities delivered

| Subsystem | What now works |
|-----------|----------------|
| Canonical JSON | Two-phase normalization (pandas/numpy/datetime/bytes/Decimal); RFC 8785 (JCS) deterministic serialization; SHA-256 stable hashes; NaN/Infinity strictly rejected; golden hash stability tests |
| Landscape audit trail | Full SQLAlchemy schema, `LandscapeRecorder`, lineage `explain()`, token-outcome recording (AUD-001), schema-compatibility checking, call-recording for external calls |
| DAG execution | `ExecutionGraph` (NetworkX-backed) with acyclicity validation, topological sort, source/sink constraints, coalesce-node creation, fork-child→coalesce linking |
| Engine | `Orchestrator` with full run lifecycle, `RowProcessor` with work-queue DAG traversal, `RetryManager` (tenacity), `AggregationExecutor` with end-of-source flush, `CoalesceExecutor`, `BatchAdapter` (passthrough + transform modes), checkpoint-after-sink-write for crash safety |
| Checkpoint / Resume | `RecoveryManager`, topology-hash validation, `NullSource` for resume, CSV-append for resume, `--execute` flag on `resume` |
| Plugin system | pluggy registration, `PluginProtocol` v1.5 (multi-row output, `creates_tokens`, `is_batch_aware`), dynamic discovery, crash-on-import-error |
| Configuration | Pydantic + Dynaconf multi-source, template-file expansion, `.env` auto-load, secret fingerprinting (HMAC, fail-closed, recursive), Azure Key Vault for fingerprint keys |
| Sources | CSVSource, JSONSource (with JSONL parse-error quarantine), NullSource, AzureBlobSource (SAS + managed identity) |
| Transforms | OpenRouterLLMTransform, AzureLLMTransform, AzureBatchLLMTransform, AzureMultiQueryLLMTransform, KeywordFilter, AzureContentSafety, AzurePromptShield, FieldMapper, Passthrough, JSONExplode, BatchReplicate, Truncate |
| LLM infrastructure | `BaseLLMTransform`, Jinja2 prompt template engine, audited LLM/HTTP clients, `CallReplayer` / `CallVerifier`, pooled execution with AIMD throttle, reorder buffer |
| Sinks | CSVSink (append-mode), JSONSink, DatabaseSink, AzureBlobSink |
| Security | HMAC secret fingerprinting (fail-closed), Azure Key Vault, multiple auth modes |
| CLI | `elspeth run --execute`, `resume`, `validate`, `plugins list`; pretty output; `.env` auto-load |
| Telemetry | Token-outcome table and recording (AUD-001) across `COMPLETED`, `ROUTED`, `FORKED`, `CONSUMED_IN_BATCH`, `COALESCED`, `QUARANTINED`, `FAILED` |
| CI / Build | Dockerfile (Python 3.12), GitHub Actions pipeline, pre-commit hooks, mutation-testing runner, line-length-140 ruff format |
| Examples | OpenRouter sentiment (standard + pooled/batched), template lookups, Azure pipelines, multi-query assessment |

### Sized

| Metric | Value |
|---|---|
| Commits | 782 |
| Calendar days | 11 |
| Active days | 11 |
| Peak day | 177 commits (2026-01-20) |
| Tests at end of period | Mutation + unit + integration regime; counts not yet pinned to a stable suite |
| Notable infrastructure | Dockerfile (Python 3.12), GitHub Actions, pre-commit hooks, mutation runner, line-length-140 ruff format |

---

<a id="period-2"></a>
## Period 2 — RC-1 Hardening (22 January – 2 February 2026)

**Outcome:** The framework moved from functional to auditable under inquiry. A telemetry subsystem was built from scratch, a chaos-testing infrastructure (ChaosLLM) was added, and over 100 bugs were triaged and closed across eight burndown sessions.

### Net additions

- **Telemetry subsystem** — event protocol, `TelemetryManager`, async export loop, `BoundedBuffer`, DROP/BLOCK back-pressure modes, `ConsoleExporter`, OTLP, Azure Monitor, Datadog (ddtrace 4.x). Orchestrator emits `RowCreated`, `TransformCompleted`, `TokenCompleted`, `RunCompleted`.
- **ChaosLLM** — fake OpenAI/Azure server with burst-state error injector, latency simulator, response generator (lorem/echo/structured-JSON), SQLite metrics recorder, MCP analysis server, pytest fixture.
- **Contract boundary hardening** — `AuditIntegrityError` + `OrchestrationInvariantError` hierarchy; `RoutingReason` discriminated union; `TransformErrorReason` TypedDict; `TokenOutcome` dataclass; `ExceptionResult` and telemetry events relocated to `contracts/` (L0).
- **Rate limiting + pooling** — two-layer rate control wired through audited clients; pool metadata integrated into the audit trail.
- **Python 3.13 in CI**; tier-model enforcement renamed and expanded.

### Sized

| Metric | Value |
|---|---|
| Commits | 811 |
| Calendar days | 11 |
| Active days | 11 |
| Peak day | 156 commits (2026-01-30) |
| Tests at end of period | Hypothesis property tests added across telemetry; ChaosLLM stress framework runnable in CI |
| Notable infrastructure | Telemetry export pipeline (OTLP, Azure Monitor, Datadog); ChaosLLM fake-server framework; rate limiting and pooling; Python 3.13 added to CI |

---

<a id="period-3"></a>
## Period 3 — RC-2 Sub-Releases (2–12 February 2026)

**Outcome:** Five rapid sub-releases delivered the architecture pieces that made the pipeline typed end-to-end, encrypted at rest, and declaratively wired.

### RC-2.0 (2 Feb)
Post-cutover cleanup: display-header support for CSV / JSON / Azure Blob sinks; `FieldResolutionApplied` telemetry event; corrupt-field-resolution crash (Tier 1 integrity); 84 stale tier-model allowlist entries pruned.

### RC-2.1 (2–3 Feb) — Key Vault Secrets + Schema Contracts
- **Azure Key Vault secrets backend** — `SecretsConfig`, `config_secrets.py` loader, new `secret_resolutions` audit table, integrated into `run` / `resume` / `validate`.
- **Schema contracts** — `FieldContract`, `SchemaContract`, `ContractBuilder` (first-row inference), contract-aware `PipelineRow` with dual-name template access, sink header modes (`contract` / `original`), contract audit columns, MCP analysis tools, checkpoint integrity verification on resume.
- **Tier 2 Langfuse tracing** — Azure, OpenRouter, and batch LLM plugins.
- **Orchestrator decomposition** — extracted into a package with `validation.py`, `export.py`, `aggregation.py`, `types.py`.

### RC-2.2 (3 Feb)
Langfuse SDK v3 (OpenTelemetry-based); failed LLM calls now produce traces.

### RC-2.3 (3–7 Feb) — PipelineRow + WebScrape + DIVERT
- **PipelineRow migration** — replaced raw `dict[str, Any]` across all signatures, executors, engine, and test suite. Checkpoint version bumped to 2.0.
- **WebScrape transform** — HTTP scrape with SSRF prevention (URL scheme validation, IP blocklist, DNS-timeout, redirect-IP TOCTOU fix), HTML extraction (markdown/text/raw), content fingerprinting, audit trail.
- **DIVERT routing** — `RoutingMode.DIVERT` edges for quarantine/error sink paths; coalesce branch-loss notification; MCP lineage annotation.
- **P0 security** — DNS-rebinding TOCTOU, atomic JSON-array sink, content-safety/prompt-shield fail-closed.
- **LandscapeRecorder** decomposed into 8 focused mixins.

### RC-2.4 (7–9 Feb) — Bug Sprint
- **178-bug triage**, 28 resolved across P0–P2 (correctness, security, engine, audit integrity).
- **Test suite v2 cutover** — 7-phase migration; **deleted 7,487 v1 tests** (222K lines); 8,138 v2 tests collected, 8,037 passing.
- Ephemeral `httpx.Client` for SSRF-safe requests (prevents TLS / SNI leakage).

### RC-2.5 (9–12 Feb) — Declarative DAG + SQLCipher + ChaosWeb
- **Declarative DAG wiring** — `WiredTransform` connection matching, every edge explicitly named and validated at construction time; processor refactored to node-ID traversal.
- **SQLCipher encryption-at-rest** — passphrase across CLI / MCP / resume; empty passphrases rejected; URI options preserved.
- **ChaosWeb** — fake web server, error injector, content generator, metrics recorder, pytest fixtures, scraping-pipeline demo (265 tests).
- **`DAGNavigator`** extracted from `RowProcessor`; executors split one-file-per-executor; MCP server split into domain modules.

### Sized

| Metric | Value |
|---|---|
| Commits | 527 |
| Calendar days | 10 |
| Active days | 10 |
| Peak day | 125 commits (2026-02-03, RC-2.1 land); 118 (2026-02-02, RC-2 cutover) |
| Tests at end of period | 8,138 v2 tests collected, 8,037 passing; v1 suite (7,487 tests) deleted at the cutover |
| Notable infrastructure | Key Vault secrets backend; schema-contract layer; declarative DAG wiring; SQLCipher; ChaosWeb fake server; LandscapeRecorder decomposed into 8 mixins; orchestrator package split |

---

<a id="period-4"></a>
## Period 4 — RC-3 Series (13 February – 10 March 2026)

**Outcome:** The system was reorganised to survive long-term maintenance. Strict typing reached the audit boundaries; the plugin tree gained an SDA-aligned shape; the LLM-transform forest was collapsed to one configurable transform; the orchestrator was split from monoliths into focused modules.

### RC-3.2 (22 Feb, tag `v0.3.0-rc3.2`)
- Strict typing at audit boundaries: `BatchCheckpointState`, `WebOutcomeClassification`, `NodeStateContext`, `CoalesceMetadata`, `AggregationCheckpointState`, `TokenUsage`, `GateEvaluationContext`, `AggregationFlushContext`, `CallPayload` protocol.
- `NodeStateGuard` enforcing terminal-state invariants in every executor.
- `detect_field_collisions()` utility preventing silent overwrites.
- Per-branch transforms between fork and coalesce nodes.
- Graceful shutdown (SIGINT / SIGTERM) for `run` and `resume`.
- Schema contracts, declarative DAG wiring, `PipelineRow`, SQLCipher, WebScrape, Langfuse v3, DIVERT — promoted from RC-2 internal hardening to public-surface stable.
- **All Alembic migrations deleted** (pre-release; no users).

### RC-3.3 (2 Mar) — Architectural Remediation
- **T10 LLM plugin consolidation** — 6 LLM transform classes (~4,950 lines) collapsed into one `LLMTransform` with provider dispatch; ~3,300 lines of duplication removed; old plugin names raise `ValueError` with migration guidance.
- **T17 PluginContext protocol split** — god-object `PluginContext` decomposed into 4 phase-based protocols (`SourceContext`, `TransformContext`, `SinkContext`, `LifecycleContext`); 23 plugin files updated.
- **T18 Orchestrator / Processor decomposition** — pure extract-method refactor; max method size ≤150 lines; typed parameter bundles (`GraphArtifacts`, `RunContext`, `LoopContext`); discriminated union types for transform / gate outcomes.
- **T19 Landscape repository pattern** — `LandscapeRecorder` from 8 mixins into 4 composed domain repositories (`RunLifecycle`, `Execution`, `DataFlow`, `Query`); recorder is now a pure delegation facade.
- **Plugins SDA restructure** — flat `plugins/` reorganised into `infrastructure/`, `sources/`, `transforms/`, `sinks/`; 247 files changed; ~200 imports rewritten.
- **Protocol relocation L3→L0** — `SourceProtocol`, `TransformProtocol`, `SinkProtocol`, `BatchTransformProtocol`, `GateResult` moved from `plugins/infrastructure/` to `contracts/`; eliminates engine→plugins layer violation.
- **ADR-006**: Layer Dependency Remediation; full architecture analysis (23 documents covering all 13 subsystems); security posture brief for v0.3.0.

### RC-3.4 (10 Mar) — Systematic Hardening
- **191-bug triage**, mutation testing, code-quality sweep — no new features; pure correctness work.
- **PayloadNotFoundError** domain exception across `PayloadStore` / `Filesystem` / `Mock` (replaces generic `KeyError`).
- **Deep-freeze utilities**: `deep_freeze()`, `deep_thaw()` recursing into tuples, frozensets, `MappingProxyType`; `slots=True` on every frozen DC; central `freeze_fields()` API.
- **Construction-time validation** — `__post_init__` on 12 dataclass types, including coalesce checkpoints (non-empty IDs, non-negative timing, disjoint branch keys).
- **Exception hygiene** — `from None` replaced with `from exc` across 16 files; 22 broad `except Exception` catches narrowed; 6 silent-skip paths converted to explicit crashes.
- **`hasattr()` banned unconditionally** (3 occurrences replaced).
- **stdlib logging → structlog** across batch mixin, multi-query, Azure-blob source, Azure batch.
- Agentic-code threat-model discussion paper (v0.1–v0.4) with MkDocs wiki and LaTeX pipeline.

### Sized

| Metric | Value |
|---|---|
| Commits | 475 |
| Calendar days | 26 |
| Active days | 25 |
| Peak day | 29 (2026-03-01), 24 (2026-03-05), 24 (2026-03-08) — RC-3.3 cutover cluster |
| Tests at end of period | ~10,563 collected at RC-3.3; +191 bugs triaged in RC-3.4 |
| Notable infrastructure | T10 / T17 / T18 / T19 architectural remediation; SDA-aligned plugin tree; deep-freeze utilities; `hasattr()` ban; structlog migration; agentic-code threat-model paper with MkDocs + LaTeX pipeline |
| Cadence note | Calmest stretch of the project at 19 commits per active day — characteristic of correctness sprints |

---

<a id="period-5"></a>
## Period 5 — RC-4 Plugins & RC-5 Cut (11 March – 3 April 2026)

> **Date note.** `CHANGELOG.md` does not stamp explicit release commits for `[0.4.0]` or `[0.4.1]`. The sub-period dates ("mid-March", "late March → 2 April") are inferred from the dominant feature-commit cluster between RC-3.4 (10 Mar) and the RC-5 cut (3 Apr): Dataverse + RAG + output-schema-contracts landed around 22–29 March; ChromaSink + `depends_on` + commencement gates landed 30 March – 2 April. The RC-5 cut date (3 April) is taken from the explicit `ELSPETH - Release Candidate 5` commit. Read each sub-period boundary as a dominant work-cluster, not a release stamp.

**Outcome:** The first external-system plugins shipped (Dataverse, RAG); the first pipeline-level orchestration primitives shipped (`depends_on`, commencement gates); the **Web UX Composer** was cut as RC-5 — a chat-first, LLM-assisted pipeline authoring surface.

### RC-4.0 (mid-March) — Plugins, Contracts, Correctness
- **Dataverse source + sink** — OData v4 REST API via `DataverseClient`, structured OData and FetchXML, pagination, SSRF validation, rate limiting. 288 new tests.
- **RAG retrieval transform** — full RAG with lifecycle management; `RetrievalProvider` protocol with `ChromaSearchProvider` (ephemeral/persistent/client modes) and `AzureSearchProvider`; three query modes (`field`, `template`, `regex`); three context modes (`numbered`, `separated`, `raw`); `PluginRetryableError` base exception.
- **Output schema contract enforcement** — `_output_schema_config` class attribute; `FrameworkBugError` if a transform declares fields but omits the schema config.
- **Audit provenance boundary** — LLM audit fields stored in `success_reason["metadata"]` instead of polluting row data; `payload_store` removed from `PluginContext`.
- **Freeze / serialize coherence** — canonical-JSON now natively handles `MappingProxyType`, `tuple`, `frozenset` in `contracts/hashing.py`.
- **CI enforcement** — `enforce_freeze_guards.py` (AST-based) detecting bare `MappingProxyType` wraps and `isinstance` guard-skips in `__post_init__`.
- **WebScrape SSRF allowlist** — three-tier validation (always-blocked → user CIDR → standard blocked); 62 new tests.
- **errorworks migration** — ChaosLLM, ChaosWeb, ChaosEngine moved to external `errorworks` PyPI package.
- **RC4-Bugsweep**: 64 bugs across 13 clusters closed.

### RC-4.1 (late March → 2 April) — RAG Ingestion Pipeline
- **ChromaSink** — write to ChromaDB with three `on_duplicate` modes (`overwrite` / `skip` / `error`); `FieldMappingConfig` (explicit, no defaults); ChromaDB metadata type validation at write time.
- **`depends_on`** — top-level config key for pre-run pipelines; `bootstrap_and_run()`; circular-dependency detection; sequential execution; `DependencyRunResult` audit DC.
- **Commencement gates** — go/no-go conditions evaluated after dependencies; uses `ExpressionParser` AST whitelist; deep-frozen `context_snapshot`; explicit `collection_probes` config.
- **Readiness contracts** — `check_readiness()` on `RetrievalProvider`; `CollectionReadinessResult`; `RetrievalNotReadyError`; `ChromaSearchProvider` and `AzureSearchProvider` implementations.
- **`preflight_results` audit table** — dependency runs and gate evaluations recorded per run.
- **End-to-end RAG example** — `examples/chroma_rag_indexed/` with indexing + query pipelines.
- **Tier 1 audit integrity hardening** — `require_int()` Tier 1 int validator (rejects Python's `bool`-is-`int` footgun); applied to 19 int fields in 13 audit DCs; TypedDict export records (15 typed shapes replacing `dict[str, Any]`); `CoalescePolicy` / `MergeStrategy` StrEnums; `Mapping[str, object]` write-path narrowing; `allow_nan=False` on 6 audit-path `json.dumps()` calls.

### RC-5 (cut 3 April) — Web UX Composer

A full web application platform for **chat-first pipeline composition**. This is the surface that turned ELSPETH from a YAML-authored framework into a system non-pipeline-engineers can drive.

| Subsystem | What now works |
|-----------|----------------|
| `elspeth web` CLI | FastAPI app factory with `[webui]` extra, `WebSettings` config, default port 8451, serves the React SPA |
| Frontend | Vite-built React SPA, `/api` and `/ws` proxy, **DTA / AGDS theming** (deep teal, green accent, GOLD), logout, session creation guards, archive sessions, confirm destructive actions, version loading, skip-to-content, reduced motion, touch-target sizing |
| Auth | `AuthProvider` protocol with `LocalAuthProvider` (bcrypt + JWT), `OIDCAuthProvider` (JWKS discovery), `EntraAuthProvider` (tenant + group claims); `get_current_user` FastAPI dependency; login / refresh / profile / config routes; configurable registration (`open` / `email_verified` / `closed`); python-jose → PyJWT migration |
| Plugin catalog | `CatalogService` protocol + REST routes |
| Sessions | SQLAlchemy Core tables + migrations; `SessionServiceImpl` with CRUD + versioning + run enforcement (`RunAlreadyActiveError`); fork-from-message; TOCTOU race elimination via DB-level constraints; thread-pool DB calls off the async loop; orphan cleanup in FastAPI lifespan |
| Blob storage | 6 phases — data model, REST API, frontend integration, composer tools, execution integration, schema inference; upload dedup; quota enforcement; file cleanup |
| Secret references | `SecretResolution` audit extension for `env` and `user` sources; `resolve_secret_refs()` tree-walk for historical `$secret{name}` references; `ServerSecretStore` + `WebSecretService` with allowlist + fingerprint audit; REST + composer + execution + frontend wiring. Current Composer authoring uses `{secret_ref: NAME}` markers for new nodes and `wire_secret_ref(...)` for existing components. |
| Execution | `ExecutionServiceImpl` with WebSocket progress; cancel-vs-execute race closure; late-WebSocket client seeding |
| Pipeline composer | Frozen `SourceSpec` / `NodeSpec` / `EdgeSpec` / `OutputSpec` / `PipelineMetadata`; composition tools + YAML generator; `ComposerService` LLM tool-use loop; sub-4x hardening (dual-counter loop guard, discovery cache, partial state recovery, rate limiting, tool registry) |
| Pipeline inspector | Inspector UX overhaul, graph readability, version selector, catalog drawer |
| Pipeline composer MCP server | `elspeth-composer` MCP server — full toolset over Model Context Protocol; pipeline-composer skill pack; wave-4 tools (`clear_source`, `explain_validation_error`, `list_models`, `preview_pipeline`) |
| Sink failsink pattern | `RowDiversion`, `SinkWriteResult`, `DIVERTED` outcome, `rows_diverted` counter, `on_write_failure` mandatory config field, `BaseSink._divert_row()`, automatic `__failsink__` DIVERT edges in DAG builder |
| DAG schema propagation | `output_schema_config` as single source of truth across source / transform / gate / aggregation / coalesce |
| Frontend UX refresh (A1–A7) | Categorized blob folders; Markdown + Mermaid rendering (with DOMPurify); route validation through chat; 50/50 panel split; secrets in toolbar; per-node validation indicators; three-state pipeline status indicator |
| Guard symmetry scanner | `enforce_guard_symmetry` CI tool — every Landscape write site must have a corresponding read guard |
| `TokenRef` type | Bundled `token_id + run_id` frozen DC; `AuditIntegrityError` loader guards; `coalesce_tokens(TokenRef)`; `_validate_token_run_ownership(TokenRef)` |
| Exception hygiene | `TIER_1_ERRORS` canonical tuple applied across all layers |
| Bug closure campaign | ~100+ P1 bugs closed plus a post-cut systematic sweep of ~130 additional bugs |
| Test hygiene | ~500 low-value tests deleted, ~200 gap-filling tests added (net: better coverage of actual behaviour) |

### Sized

| Metric | Value |
|---|---|
| Commits | 478 (across the three RC-4 + RC-5 cut sub-periods) |
| Calendar days | 24 |
| Active days | 21 |
| Peak day | Various; smaller bursts rather than single peaks |
| Tests at end of period | ~10,500 framework tests plus composer + web track additions |
| Notable infrastructure | Dataverse OData v4 client; RAG retrieval contract; ChromaSink; `depends_on` orchestration; FastAPI web app; React SPA with AGDS theming; three-provider auth; composer LLM tool-use loop; MCP composer server |

---

<a id="period-6"></a>
## Period 6 — RC-5.1 Composer Correctness (4 April – 11 May 2026)

**Outcome:** RC-5 had shipped the surface; RC-5.1 made it **answer accurately**. The composer's authoring loop, validation surface, and run-evidence views all received targeted hardening; the run-evidence story was extended with discard summaries, failure-sample aggregation, and a cancellation-requested badge. The most consequential fix in this period closed a **Tier-1 wire-visible-fabrication defect**: during a development-environment evaluation run, the composer's LLM had copied a placeholder example value (`compliance@example.com`, supplied in a skill-prompt code snippet) into the `web_scrape.http.abuse_contact` HTTP header on requests to three Australian Government websites. The defect was detected by ELSPETH's own audit trail (the evaluation harness, not production traffic), and the audit record is the reason this incident is documented here. The remediation — `<OPERATOR_REQUIRED>` sentinels for identity-bearing fields, a hard rule against silent operator-input rewrites, and an implicit-decision disclosure block — is described under *Validator and Pipeline Authoring (RC-5.1)* below. No production deployment was affected.

### Surface expansion

- **Substrate-first README** — ELSPETH framed as a high-assurance pipeline substrate with two authoring surfaces (hand-edited YAML for operators, Web Composer for LLM-assisted authoring by non-pipeline-engineers); validator-mediated authoring framing; expanded composer surface description; audited composer tool loop; runtime-shaped validation and preflight; run-evidence endpoints; cancellation visibility.
- **Composer reliability + operator visibility** — deterministic composer calls (temperature/seed in audit sidecar); prompt-cache-aware audit; reasoning metadata capture; **advisor escalation contract** (frontier-model escalation gated behind mechanically validated trigger categories); hard-mode evaluation harness; advisor-conditional skill markers (`<!-- ADVISOR-ONLY -->` / `<!-- ADVISOR-DISABLED -->`).
- **Plugin / contract surface** — plugins can publish semantic facts, requirements, comparison outcomes, and composer assistance text; **10 statistical batch plugins** added with runnable examples (`batch_distribution_profile`, `batch_experiment_compare`, `batch_classifier_metrics`, `batch_paired_preference`, `batch_drift_compare`, `batch_outlier_annotator`, `batch_data_quality_report`, `batch_top_k`, `batch_threshold_summary`, `batch_effect_size`); richer missing-dependency / schema vocabulary / repair hints.
- **Two-axis terminal outcome model** — lifecycle outcome separated from terminal path/provenance; run accounting split into routed-success/failure, token-lifecycle counts, source-row counts, discard summaries, closure integrity.
- **CI / policy / test surface** — gates added or tightened for component types, guard symmetry, audit-evidence nominal typing, Tier-1 decoration, contract manifests, composer exception channels, composer catch order, tier-model allowlists; **Frontend Playwright baseline** booting FastAPI + Vite for E2E.

### RC-5.1 specifics (composer correctness)

- **`identity_node_advisory` validator** — `validate_pipeline` detects identity passthrough nodes silently degrading observed-sink lineage; gated by an exemption matrix.
- **Composer pipeline recipes** — `apply_pipeline_recipe` MCP tool + two initial templates; deep-frozen `RecipeSpec.slots`; recipes emit schema-valid `llm` and `type_coerce` options.
- **Source inspection MCP tool** — `inspect_source` surfaces external-data shape and silent-failure modes (e.g. all rows quarantined) as warnings.
- **Forced-repair loop with proof diagnostics** — `preview_pipeline` runs a proof step; `compute_proof_diagnostics` verifies blob `content_hash` (stale blob can no longer pass the proof gate); `_BLOCKING_DIAGNOSTIC_CODES` is the structural source of truth.
- **Audit-backend skill + recipe-first fork-coalesce shape** in composer skill pack; mandatory advisor-escalation gate for Recipe #10 (fork+coalesce).
- **Convergence-suite scenarios** — URL-text smoke, mocked-LLM integration, end-to-end forced-repair, end-to-end `apply_pipeline_recipe`, fork-and-coalesce regression.
- **Composer authoring affordances** — "Use in pipeline" prefills chat input; chat code blocks gain syntax highlight + copy-to-clipboard; `secret_ref` inline form; resize-handle keyboard arrows; touch-friendly hit zone.
- **`<OPERATOR_REQUIRED>` sentinels** — replace literal example values for identity-bearing fields (`web_scrape.http.abuse_contact`, `scraping_reason`); explicit resolution order (operator-supplied → deployment-identity → ask before `set_pipeline`).
- **Hard rule against silent operator-input rewrites** — any normalisation must be confirmed or routed through a recorded step that appears in YAML.
- **Implicit-decision disclosure** ("Decisions I made on your behalf") — Build Summary now enumerates operator-invisible authoring decisions with provenance markers (`default` / `picked` / `deployment-identity` / `operator-supplied`).
- **Run-evidence widgets** — `RunOutputsPanel` (full audit-evidence manifest, downloadable artifacts gated by `downloadable` flag); cancellation-requested badge separate from terminal `cancelled`; GraphView viewport preservation across topology changes; `data_dir` resolved to absolute path at validation time; source-inspection silent-failure surfacing; failure-sample aggregation in run-level errors.
- **Audit-integrity test coverage** — direct unit coverage added for **ADR-019 deferred-invariant sweep**, `_validate_token_row_ownership`, `link_validation_error_to_row` branches, all 12 entries in `_REQUIRED_COMPOSITE_FOREIGN_KEYS`, and SSRF blocked-IP residual coverage (including `::ffff:0:0/96` and seven other previously untested boundary cases).
- **Tier-1 panel-review accessibility pass** — `aria-controls` IDREF when run collapsed; `aria-expanded` on `RunsView` Inspect; health-check banners downgraded to `role=status`; nested `aria-live` removed from `ComposingIndicator`; light-theme `--color-status-empty` override.

### Sized

| Metric | Value |
|---|---|
| Commits | 810 |
| Calendar days | 38 |
| Active days | 37 |
| Peak day | 93 (2026-05-09) — Phase 1B persist_compose_turn happy path |
| Tests at end of period | ~10,500 framework + composer/web additions; audit-integrity test suite expanded for ADR-019 deferred-invariant sweep |
| Notable infrastructure | Substrate-first README; composer reliability + advisor escalation; 10 statistical batch plugins; two-axis terminal outcome model; identity-node-advisory validator; OPERATOR_REQUIRED sentinels; forced-repair loop with proof diagnostics; Playwright E2E baseline |

---

<a id="period-7"></a>
## Period 7 — RC-5.2 Guided Composer + Recovery (12–19 May 2026)

**Outcome:** The Web Composer became an audited, recoverable authoring system. Model calls, tool dispatches, redacted tool payloads, persisted transcript rows, recovery diffs, and operator-visible failure causes are now recorded against one shared evidence model.

### Composer Guided Mode

- **Guided wizard** — structured-protocol pipeline authoring for first-time users; source → sink → transforms in three steps; closed six-turn taxonomy; deterministic recipe pre-match; LLM is read-only with respect to pipeline state. Ships alongside the unmodified freeform composer; mode transition uses progressive disclosure.
- **`ComposerLLMCall` audit channel** — every `solve_chain` invocation records a row (provider, model, status, latency, prompt/completion tokens). Pairs with the existing `ComposerToolInvocation` audit channel.

### Composer Progress Persistence (Phases 1A → 4)

- **Phase 1A — schema** — new `chat_messages` audit columns (`tool_call_id`, `sequence_no`, `writer_principal`, `parent_assistant_id`) with biconditional CHECK constraints pinning OpenAI-shaped tool-call linkage; `composition_states.provenance` enum (`tool_call` / `convergence_persist` / `plugin_crash_persist` / `preflight_persist` / `session_seed` / `session_fork`); `run_events` table; `audit_access_log` table (INERT for Phase 3+); per-session indices.
- **Phase 1B — single-transaction primitive** — `SessionServiceImpl.persist_compose_turn` (sync + async); `StatePayload` / `_ToolOutcome` / `RedactedToolRow` / `AuditOutcome` DTOs; advisory-lock primitive (`pg_advisory_xact_lock` on Postgres, per-session `RLock` on SQLite); sequence-number reservation under session write lock.
- **Phase 1C — Postgres portability** — `@pytest.mark.testcontainer` lane spinning up ephemeral Postgres per test; exercises `pg_advisory_xact_lock`, commit-wins concurrency, and Postgres-specific blob `ready_hash` partial uniqueness.
- **Phase 2 — redaction walker + MANIFEST** — `web/composer/redaction.py` grew from a 42-line stub to a **2,752-line walker**; 38-entry MANIFEST (10 type-driven + 28 declarative) with Pydantic argument models; Pydantic-first ARG_ERROR routing (`ToolArgumentError` re-raise; LLM-facing message names argument-bundle only — rev-2 BLOCKER_A leak discipline); structured Pydantic detail preserved on `__cause__` via `canonicalize_pydantic_cause`; adequacy guard pinning manifest-registry parity and a byte-identical redaction snapshot; F1–F6 hardening (completeness Hypothesis property tests, walker-guard parity, summarizer-contract Hypothesis, label-gate CI workflow, drift guards).
- **Phase 3 — compose-loop persistence** — `_compose_loop` persists assistant messages, tool-call breadcrumbs, redacted tool payloads, and composition-state snapshots through `persist_compose_turn`; audit-first contract preserved through tool failure, mid-turn cancellation, and plugin-crash recovery; per-turn tool-call cap with `tool_call_cap_exceeded` reason code; `include_tool_rows=true` opt-in for audit-grade transcript access recorded through `audit_access_log`; property + integration tests for audit counter conservation, manifest redaction, cancellation commit windows, failed-turn tool-response counts, no-op behaviour.
- **Phase 4 — frontend recovery** — recovery panel detecting recoverable composer failures; assistant transcript + redacted tool rows + before/after state diff rendered via `RecoveryTranscript` / `RecoveryDiff` / `RecoveryPanel`; frontend `npm run lint` gate added.

### RC-5.2 Hotfix Integration (folded back to `main`)

- **Auth + audit hardening** — local/Entra audit token issuance, auth-failure classes, local login outcomes, refresh-provider invariants, provider outages, web-run attribution to Landscape; JWKS-failure detail redacted; token-response caching suppressed.
- **Execution + validation hardening** — web execution classifies validation errors, sanitises broad execution errors, persists resolved run config, rejects misplaced secret refs, preserves guided audit persistence failures.
- **Engine / plugin correctness fixes** — checkpoint resume parsing; empty coalesce checkpoint state; pending batch row identities; JSON sink parent creation; sink preflight collision timing; Web Scrape fail-closed boundaries; LLM provider preflight; shared LLM telemetry helpers.
- **Frontend accessibility + theming fixes** — improved contrast on guided/catalog/run UI; forced-colors fallback; theme initialisation + cross-tab sync; screen-reader-safe status symbols; catalog retry controls; keyboard shortcut support; preserved plugin descriptions.

### Phase 6 / 7 / 8 (post-5.2 merge stream)

- **Phase 5** — chat-data-entry: dynamic 1-row source from chat text for hello-world / canonical-test-case shapes.
- **Phase 6** — completion gestures: Save-for-review / Run-analysis / Execute / Copy-YAML differentiated by persona.
- **Phase 7** — catalog reshape (16 / 16a / 16b / 16c batched): drawer reframed from interactive toolkit to searchable reference; plugin-coverage gates calibrated for `report_assemble`; `i1`/`i2`/`i3` fixes (drawer error log, snapshot lock, NETWORK retirement).
- **Phase 8** — final polish sweep: dead-code removal, lint, CSS tokens; four contrast-scan surfaces enumerated for Phase 9; CICD allowlist burn-down merged.

### Operational change of note

> Sessions DB schema deployment requires recreation. Phase 1A columns / tables / CHECKs / partial unique indices are **not applied via Alembic**. Operator stops the service, archives the old `sessions.db`, and restarts; the bootstrap creates the new schema on first start. Procedure documented in `docs/runbooks/staging-session-db-recreation.md`.

### Sized

| Metric | Value |
|---|---|
| Commits | 638 (12 → 19 May; count frozen for this report on 2026-05-19) |
| Calendar days | 8 |
| Active days | 8 |
| Peak day | 142 (2026-05-12) — RC-5.2 composer redaction MANIFEST pass; 97 (2026-05-14) RC-5.2 release-stamp; 94 (2026-05-18, 2026-05-19) |
| Tests at end of period | 3,159 backend + 447 frontend (per-step-chat branch baseline); framework suite continues at ~10,500 |
| Notable infrastructure | Composer guided mode; `ComposerLLMCall` audit channel; 4-phase composer progress persistence (schema → primitive → Postgres portability → redaction walker → compose-loop persistence → frontend recovery); 2,752-line redaction walker; Postgres testcontainer lane; recovery panel UX |
| Cadence note | Highest sustained tempo since RC-1 cutover (79.8 commits / active day over 8 consecutive days) |

---

## Cumulative Output Snapshot

### Code

| Measure | RC-1 | RC-2 (0.1.0) | RC-3.4 | RC-5.0 | Current (RC-5.2 + Phase 8) |
|---------|-----:|------------:|-------:|-------:|---------------------------:|
| Commits to date | 782 | 1,593 | 2,595 | 3,073 | 4,521 |
| Framework test suite | mutation + unit + integration (Phase 1–3 regime) | 8,138 collected (v2) | ~10,500 | ~10,500 + composer/web additions | ~10,500 framework |
| Composer-track tests (RC-5.2) | — | — | — | — | 3,159 backend + 447 frontend (per-step-chat branch baseline) |
| Major plugin surface | 13+ plugins (sources / transforms / sinks) | + ChaosLLM | + LLM consolidation (1 transform, dispatch) | + Dataverse, RAG, ChromaSink | + 10 statistical batch plugins, source-inspection, recipes |

### Capabilities

| Capability | First shipped |
|-----------|---------------|
| Auditable SDA pipeline | RC-1 |
| LLM transforms (Azure, OpenRouter, batch, multi-query) | RC-1 |
| Crash-safe checkpoint / resume | RC-1 |
| Telemetry subsystem (OTLP, Azure Monitor, Datadog) | RC-1 Hardening (pre-RC-2) |
| ChaosLLM stress / fault injection | RC-1 Hardening |
| Azure Key Vault secrets backend | RC-2.1 |
| Schema contracts (first-row inference, propagation, audit) | RC-2.1 |
| Tier 2 Langfuse tracing (v3) | RC-2.1 / 2.2 |
| Typed `PipelineRow` end-to-end | RC-2.3 |
| WebScrape transform with SSRF prevention | RC-2.3 |
| DIVERT routing for quarantine/error sinks | RC-2.3 |
| SQLCipher encryption-at-rest | RC-2.5 |
| Declarative DAG wiring | RC-2.5 |
| ChaosWeb fake server | RC-2.5 |
| Strict typing at audit boundaries | RC-3.2 |
| Single configurable LLM transform (T10) | RC-3.3 |
| Layer-enforced 4-tier architecture (L0–L3) | RC-3.3 |
| Repository pattern for Landscape | RC-3.3 |
| Deep-immutability `freeze_fields` API | RC-3.4 |
| `hasattr()` ban | RC-3.4 |
| Dataverse plugin (OData v4) | RC-4.0 |
| RAG retrieval transform | RC-4.0 |
| Output schema contract enforcement | RC-4.0 |
| `enforce_freeze_guards.py` CI gate | RC-4.0 |
| `errorworks` external PyPI package | RC-4.0 |
| ChromaSink (vector store population) | RC-4.1 |
| `depends_on` pipeline sequencing | RC-4.1 |
| Commencement gates | RC-4.1 |
| Readiness contracts on retrieval providers | RC-4.1 |
| **`elspeth web` CLI + React composer** | RC-5.0 |
| Three-provider auth (Local / OIDC / Entra) | RC-5.0 |
| Blob storage manager + secret references | RC-5.0 |
| Background pipeline execution with WebSocket progress | RC-5.0 |
| Sink failsink pattern | RC-5.0 |
| `elspeth-composer` MCP server | RC-5.0 |
| `TokenRef` type | RC-5.0 |
| Guard-symmetry CI scanner | RC-5.0 |
| Frontend Playwright E2E | RC-5.1 |
| Composer pipeline recipes | RC-5.1 |
| Source-inspection MCP tool | RC-5.1 |
| Forced-repair loop with proof diagnostics | RC-5.1 |
| 10 statistical batch plugins | RC-5.1 |
| Composer guided mode | RC-5.2 |
| `ComposerLLMCall` audit channel | RC-5.2 |
| Composer progress persistence (4 phases) | RC-5.2 |
| 38-entry redaction MANIFEST + 2,752-line walker | RC-5.2 |
| Postgres-portability testcontainer lane | RC-5.2 |
| Frontend recovery UX (panel + diff + redacted tool rows) | RC-5.2 |
| Per-step guided chat | RC-5.2 |

### Process / Discipline

| Discipline | First enforced |
|-----------|----------------|
| Canonical JSON via RFC 8785 + SHA-256 | RC-1 |
| Three-tier trust model (Tier 1 / Tier 2 / Tier 3) | RC-1 |
| `frozen=True` + `slots=True` on audit DTOs | RC-3.2 / 3.3 |
| `enforce_tier_model.py` CI gate | RC-1 Hardening |
| `enforce_freeze_guards.py` CI gate | RC-4.0 |
| `enforce_guard_symmetry` CI gate | RC-5.0 |
| Layer-import enforcement (L0→L1→L2→L3) | RC-3.3 |
| Test factory architecture (`make_context`, `make_recorder_with_run`) | RC-3.3 |
| 4-layer model + ADR-006 | RC-3.3 |
| Property-based testing (Hypothesis) across SSRF, ChaosLLM, DAG, triggers, routing, schema contracts, reorder buffer, orchestrator lifecycle | RC-2.4 onwards |
| Mutation-survivor regimen (>71 new tests killed in RC-4.0 alone) | RC-3.3 onwards |
| `hasattr()` banned unconditionally | RC-3.4 |
| `TIER_1_ERRORS` canonical exception tuple | RC-5.0 |
| Composer audit-MANIFEST adequacy guard + byte-identical snapshot | RC-5.2 |
| CICD-allowlist-audit skill (periodic 4–8 week pass) | RC-5.2 |

---

## Post-RC-5.2 follow-up themes

The following themes are visible from the release evidence and remain useful
planning inputs after RC-5.2:

- **Composer correctness cluster** — validator parity, runtime dry-run, operator visibility (elspeth-528bde62bb).
- **Fork / coalesce audit-integrity epic** — schema reconciliation, field provenance, merge safety (elspeth-e20903300c).
- **Web auth hardening** (OIDC / Entra / JWKS) (elspeth-250f698aaf).
- **Web sessions + Alembic-env reconciliation** (elspeth-ef52049338).
- **Plugin Expansion Phase 1** — web research pipeline (OpenSearch, browser scrape, report sink, Chroma upgrade) (elspeth-868c55d712).
- **Composer progress persistence** — tool-call breadcrumbs and partial drafts surviving long-running failures (elspeth-90b4542b63).

---

## What this report does not cover

This is the project-level outputs view. For depth on the topics it summarises:

- **Plugin internals** — see the README of each plugin directory under `src/elspeth/plugins/`.
- **Frontend internals** — see `src/elspeth/web/frontend/` (components, stores, tests).
- **Pipeline-author UX** — see the `pipeline-composer` skill and its companion sessions under `evals/composer-rgr/`.
- **Operational procedures** — see `docs/runbooks/` (resume, routing investigation, incident response, database maintenance, backup, Key Vault configuration, Ansible-based Ubuntu deployment).
- **ADRs (architectural decisions)** — see `docs/architecture/adr/`.
- **The contractual surface** — see `guarantees.md` in this directory.
- **Per-day commit cadence** — see `elspeth-velocity-rc1-to-rc5.md` in this directory.
- **Audience-tier executive view** — see `executive-summary.md` in this directory (drafted for Australian public-sector evaluators).

---

## Sources

- `../../docs-archive/2026-05-19-docs-cleanout/CHANGELOG-RC1.md` — Pre-RC1 and RC-1 Hardening (Jan 12 – Feb 2, 2026) — archived snapshot
- `../../docs-archive/2026-05-19-docs-cleanout/CHANGELOG-RC2.md` — RC-2.0 through RC-2.5 (Feb 2–12, 2026) — archived snapshot
- `/CHANGELOG.md` — active release changelog from RC-3.2 through RC-5.2 + Unreleased (Feb 13, 2026 – present)
- `../../docs-archive/2026-05-19-docs-cleanout/docs/release/feature-inventory.md` — RC-3.3 feature reconciliation (1 March 2026) — archived snapshot
- `../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc4-executive-brief.md` — RC-4 work-package summary (3 March 2026) — archived planning brief
- Git history snapshots for RC-1, RC-2, RC-5.2, and `main`
- Maintainer planning snapshot (19 May 2026)
