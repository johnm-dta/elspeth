// progress.typ — ELSPETH Progress Report, RC-1 to RC-5
// Audience: engineering reviewers. Sober and readable.

#import "tokens.typ": *
#import "theme.typ": *
#import "data.typ" as d
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": chart, plot

#show: document-frame.with(
  title: "ELSPETH Progress Report",
  subtitle: "Engineering progress, RC-1 to RC-5",
  draft: false,
  h1-pagebreak: false,
)

#cover-page(
  title: "Progress Report",
  subtitle: "Engineering progress, RC-1 to RC-5.",
  doc-date: d.doc-date,
  version: "RC-5.2",
  author: "John Morrissey, CTO Branch",
  affiliation: "Digital Transformation Agency",
  audience: "DTA Architecture, Security and Technical staff",
  hero: cover-hero-sda(),
)

= Period and scope

// Stripe colours rotate semantic registers across the headline cards:
//   period      -> c-action     (informational)
//   commits     -> c-supported  (positive output)
//   active days -> c-navy-soft  (structural fact)
#grid(columns: (1fr, 1fr, 1fr), gutter: sp-3,
  metric-card("Period covered", "128 days",
    sub: "12 January -- 19 May 2026",
    colour: c-action),
  metric-card("Cumulative commits", "4,521",
    sub: "Unique across RC history snapshots",
    colour: c-supported),
  metric-card("Active commit days", "123",
    sub: "5 idle days (post-release pauses)",
    colour: c-navy-soft),
)

#v(sp-3)

The repository state at 19 May 2026 holds *4,521 unique commits* across
the RC-1, RC-2, and RC-5.2 history snapshots. This document gives the
*progress / outputs* view — cumulative capability and per-period
engineering work. For per-day work volume, see the companion
#emph[Velocity] document.

= Executive timeline

The thirteen release milestones below trace the project from empty
scaffold to a web-authored, audit-grounded pipeline platform.
Cumulative commit counts at each milestone come from the canonical
RC commit message, the version-bump commit, or the release tag.

#data-table(
  columns: 4,
  header: ([Release], [Date], [Theme], [Cumulative]),
  align-rules: (left + horizon, left + horizon, left + horizon,
    right + horizon),
  ..d.release-milestones.map(((rel, date, theme, n)) => (
    [#rel], [#date], [#theme], [#str(n)],
  )),
)

#v(sp-3)

#callout(kind: "advisory", title: "Date qualification — RC-4.0 and RC-4.1")[
  `CHANGELOG.md` does not stamp explicit release commits for `[0.4.0]`
  or `[0.4.1]`. The date ranges are inferred from the dominant
  feature-commit clusters between RC-3.4 (10 Mar) and the RC-5 cut
  (3 Apr): Dataverse + RAG + output-schema-contracts landed around
  22--29 March; ChromaSink + `depends_on` + commencement gates landed
  30 March -- 2 April. Read the boundary as the dominant work-cluster,
  not a release stamp.
]

= Cumulative commit growth

#chart-figure(
  caption: [Cumulative commits at each release milestone, from project
    start (12 January 2026) to the 19 May 2026 release snapshot. The slope
    flattens during the RC-3 series (architectural remediation,
    smaller per-commit footprint) and re-accelerates during RC-5.2
    composer maturation.],
  description: [Line chart of cumulative commits over time. The line
    rises steeply from 0 in mid-January to 1,593 by RC-2 in early
    February, plateaus through February-March around 2,000-2,600,
    re-accelerates through April to 3,073 at RC-5, and rises steeply
    again to 4,521 at the 19 May snapshot.],
  data: data-table(
    header: ([Release], [Date], [Cumulative commits]),
    align-rules: (left + horizon, left + horizon, right + horizon),
    ..d.release-milestones.map(((rel, date, theme, n)) => (
      [#rel], [#date], [#str(n)],
    )),
  ),
  cetz.canvas({
    // Short labels: strip the "(0.x.y)" parenthetical and "Project " /
    // "Current " prefixes so x-axis ticks don't overlap.
    let short-label(s) = {
      let t = s.replace(regex("\\s*\\(.+?\\)"), "")
      t = t.replace("Project start", "Start")
      t = t.replace("RC-5.2 snapshot", "RC-5.2")
      t
    }
    let pts = d.release-milestones.enumerate().map(((i, item)) => {
      (i, item.at(3))
    })
    plot.plot(
      size: (15, 6),
      x-tick-step: none,
      x-ticks: d.release-milestones.enumerate().map(((i, item)) => {
        (i, short-label(item.at(0)))
      }),
      x-min: -0.3, x-max: pts.len() - 0.7,
      y-tick-step: 1000,
      y-min: 0, y-max: 5000,
      x-label: none,
      y-label: "Cumulative commits",
      {
        plot.add(pts, line: "linear", style: (stroke: 2pt + c-action))
        plot.add(pts, mark: "o", mark-size: 0.18,
          mark-style: (fill: c-navy, stroke: 1pt + c-navy),
          style: (stroke: none))
      },
    )
  }),
)

= Period 1 — Foundation (Pre-RC1, 12--22 January 2026)

#callout(kind: "success", title: "Outcome")[
  ELSPETH went from an empty scaffold to a working auditable
  Sense/Decide/Act pipeline framework with CLI, plugin system, full
  DAG executor, LLM integration, Azure ecosystem support, and
  crash-safe recovery — in *eleven days*.
]

== Capabilities delivered

#data-table(
  columns: 2,
  header: ([Subsystem], [What now works]),
  align-rules: (left + horizon, left + horizon),
  [Canonical JSON],
  [Two-phase normalization (pandas/numpy/datetime/bytes/Decimal); RFC 8785 (JCS) deterministic serialization; SHA-256 stable hashes; NaN/Infinity strictly rejected; golden hash stability tests],
  [Landscape audit trail],
  [Full SQLAlchemy schema, `LandscapeRecorder`, lineage `explain()`, token-outcome recording (AUD-001), schema-compatibility checking, call-recording for external calls],
  [DAG execution],
  [`ExecutionGraph` (NetworkX-backed) with acyclicity validation, topological sort, source/sink constraints, coalesce-node creation, fork-child to coalesce linking],
  [Engine],
  [`Orchestrator` with full run lifecycle, `RowProcessor` with work-queue DAG traversal, `RetryManager` (tenacity), `AggregationExecutor` with end-of-source flush, `CoalesceExecutor`, `BatchAdapter`, checkpoint-after-sink-write for crash safety],
  [Checkpoint / Resume],
  [`RecoveryManager`, topology-hash validation, `NullSource` for resume, CSV-append for resume, `--execute` flag on `resume`],
  [Plugin system],
  [pluggy registration, `PluginProtocol` v1.5 (multi-row output, `creates_tokens`, `is_batch_aware`), dynamic discovery, crash-on-import-error],
  [Configuration],
  [Pydantic + Dynaconf multi-source, template-file expansion, `.env` auto-load, secret fingerprinting (HMAC, fail-closed, recursive), Azure Key Vault for fingerprint keys],
  [Sources],
  [CSVSource, JSONSource (with JSONL parse-error quarantine), NullSource, AzureBlobSource (SAS + managed identity)],
  [Transforms],
  [OpenRouterLLMTransform, AzureLLMTransform, AzureBatchLLMTransform, AzureMultiQueryLLMTransform, KeywordFilter, AzureContentSafety, AzurePromptShield, FieldMapper, Passthrough, JSONExplode, BatchReplicate, Truncate],
  [LLM infrastructure],
  [`BaseLLMTransform`, Jinja2 prompt template engine, audited LLM/HTTP clients, `CallReplayer` / `CallVerifier`, pooled execution with AIMD throttle, reorder buffer],
  [Sinks],
  [CSVSink (append-mode), JSONSink, DatabaseSink, AzureBlobSink],
  [Security],
  [HMAC secret fingerprinting (fail-closed), Azure Key Vault, multiple auth modes],
  [CLI],
  [`elspeth run --execute`, `resume`, `validate`, `plugins list`; pretty output; `.env` auto-load],
  [Telemetry],
  [Token-outcome table and recording (AUD-001) across `COMPLETED`, `ROUTED`, `FORKED`, `CONSUMED_IN_BATCH`, `COALESCED`, `QUARANTINED`, `FAILED`],
  [CI / Build],
  [Dockerfile (Python 3.12), GitHub Actions pipeline, pre-commit hooks, mutation-testing runner, line-length-140 ruff format],
  [Examples],
  [OpenRouter sentiment (standard + pooled/batched), template lookups, Azure pipelines, multi-query assessment],
)

== Sized

- *782 commits* in 11 calendar days (average 71/day; peak 177 on 2026-01-20).
- Comprehensive test regime across unit, integration, and mutation testing — golden-hash stability tests, contract tests for every plugin, property-based tests for the reorder buffer.

= Period 2 — RC-1 Hardening (22 January -- 2 February 2026)

#callout(kind: "success", title: "Outcome")[
  The framework moved from "works" to "withstands inquiry". A telemetry
  subsystem was built from scratch; a chaos-testing infrastructure
  (ChaosLLM) was added; over 100 bugs were triaged and closed across
  eight burndown sessions.
]

== Net additions

- *Telemetry subsystem* — event protocol, `TelemetryManager`, async export loop, `BoundedBuffer`, DROP/BLOCK back-pressure modes, `ConsoleExporter`, OTLP, Azure Monitor, Datadog (ddtrace 4.x). Orchestrator emits `RowCreated`, `TransformCompleted`, `TokenCompleted`, `RunCompleted`.
- *ChaosLLM* — fake OpenAI/Azure server with burst-state error injector, latency simulator, response generator (lorem/echo/structured-JSON), SQLite metrics recorder, MCP analysis server, pytest fixture.
- *Contract boundary hardening* — `AuditIntegrityError` + `OrchestrationInvariantError` hierarchy; `RoutingReason` discriminated union; `TransformErrorReason` TypedDict; `TokenOutcome` dataclass; `ExceptionResult` and telemetry events relocated to `contracts/` (L0).
- *Rate limiting + pooling* — two-layer rate control wired through audited clients; pool metadata integrated into the audit trail.
- *Python 3.13 in CI*; tier-model enforcement renamed and expanded.

== Sized

- *811 commits* in 11 calendar days (average 74/day; peak 156 on 2026-01-30).
- Hypothesis property tests added across telemetry; ChaosLLM stress framework runnable in CI.

= Period 3 — RC-2 Sub-Releases (2--12 February 2026)

#callout(kind: "success", title: "Outcome")[
  Five rapid sub-releases delivered the architecture pieces that made
  the pipeline *typed end-to-end*, *encrypted at rest*, and
  *declaratively wired*.
]

== RC-2.0 (2 Feb)
Post-cutover cleanup: display-header support for CSV / JSON / Azure Blob sinks; `FieldResolutionApplied` telemetry event; corrupt-field-resolution crash (Tier 1 integrity); 84 stale tier-model allowlist entries pruned.

== RC-2.1 (2--3 Feb) — Key Vault Secrets + Schema Contracts
- *Azure Key Vault secrets backend* — `SecretsConfig`, `config_secrets.py` loader, new `secret_resolutions` audit table, integrated into `run` / `resume` / `validate`.
- *Schema contracts* — `FieldContract`, `SchemaContract`, `ContractBuilder` (first-row inference), contract-aware `PipelineRow` with dual-name template access, sink header modes (`contract` / `original`), contract audit columns, MCP analysis tools, checkpoint integrity verification on resume.
- *Tier 2 Langfuse tracing* — Azure, OpenRouter, and batch LLM plugins.
- *Orchestrator decomposition* — extracted into a package with `validation.py`, `export.py`, `aggregation.py`, `types.py`.

== RC-2.2 (3 Feb)
Langfuse SDK v3 (OpenTelemetry-based); failed LLM calls now produce traces.

== RC-2.3 (3--7 Feb) — PipelineRow + WebScrape + DIVERT
- *PipelineRow migration* — replaced raw `dict[str, Any]` across all signatures, executors, engine, and test suite. Checkpoint version bumped to 2.0.
- *WebScrape transform* — HTTP scrape with SSRF prevention (URL scheme validation, IP blocklist, DNS-timeout, redirect-IP TOCTOU fix), HTML extraction (markdown/text/raw), content fingerprinting, audit trail.
- *DIVERT routing* — `RoutingMode.DIVERT` edges for quarantine/error sink paths; coalesce branch-loss notification; MCP lineage annotation.
- *P0 security* — DNS-rebinding TOCTOU, atomic JSON-array sink, content-safety/prompt-shield fail-closed.
- *LandscapeRecorder* decomposed into 8 focused mixins.

== RC-2.4 (7--9 Feb) — Bug Sprint
- *178-bug triage*, 28 resolved across P0--P2 (correctness, security, engine, audit integrity).
- *Test suite v2 cutover* — 7-phase migration; *deleted 7,487 v1 tests* (222K lines); 8,138 v2 tests collected, 8,037 passing.
- Ephemeral `httpx.Client` for SSRF-safe requests (prevents TLS / SNI leakage).

== RC-2.5 (9--12 Feb) — Declarative DAG + SQLCipher + ChaosWeb
- *Declarative DAG wiring* — `WiredTransform` connection matching, every edge explicitly named and validated at construction time; processor refactored to node-ID traversal.
- *SQLCipher encryption-at-rest* — passphrase across CLI / MCP / resume; empty passphrases rejected; URI options preserved.
- *ChaosWeb* — fake web server, error injector, content generator, metrics recorder, pytest fixtures, scraping-pipeline demo (265 tests).
- *`DAGNavigator`* extracted from `RowProcessor`; executors split one-file-per-executor; MCP server split into domain modules.

== Sized

- *527 commits* in 10 days. Peak 125 on 2026-02-03 (RC-2.1 land); 118 on 2026-02-02 (RC-2 cutover).

= Period 4 — RC-3 Series (13 February -- 10 March 2026)

#callout(kind: "success", title: "Outcome")[
  The system was reorganised to survive long-term maintenance. Strict
  typing reached the audit boundaries; the plugin tree gained an
  SDA-aligned shape; the LLM-transform forest was collapsed to one
  configurable transform; the orchestrator was split from monoliths
  into focused modules.
]

== RC-3.2 (22 Feb, tag `v0.3.0-rc3.2`)
- Strict typing at audit boundaries: `BatchCheckpointState`, `WebOutcomeClassification`, `NodeStateContext`, `CoalesceMetadata`, `AggregationCheckpointState`, `TokenUsage`, `GateEvaluationContext`, `AggregationFlushContext`, `CallPayload` protocol.
- `NodeStateGuard` enforcing terminal-state invariants in every executor.
- `detect_field_collisions()` utility preventing silent overwrites.
- Per-branch transforms between fork and coalesce nodes.
- Graceful shutdown (SIGINT / SIGTERM) for `run` and `resume`.
- Schema contracts, declarative DAG wiring, `PipelineRow`, SQLCipher, WebScrape, Langfuse v3, DIVERT — promoted from RC-2 internal hardening to public-surface stable.
- *All Alembic migrations deleted* (pre-release; no users).

== RC-3.3 (2 Mar) — Architectural Remediation
- *T10 LLM plugin consolidation* — 6 LLM transform classes (about 4,950 lines) collapsed into one `LLMTransform` with provider dispatch; about 3,300 lines of duplication removed; old plugin names raise `ValueError` with migration guidance.
- *T17 PluginContext protocol split* — god-object `PluginContext` decomposed into 4 phase-based protocols (`SourceContext`, `TransformContext`, `SinkContext`, `LifecycleContext`); 23 plugin files updated.
- *T18 Orchestrator / Processor decomposition* — pure extract-method refactor; max method size <= 150 lines; typed parameter bundles (`GraphArtifacts`, `RunContext`, `LoopContext`); discriminated union types for transform / gate outcomes.
- *T19 Landscape repository pattern* — `LandscapeRecorder` from 8 mixins into 4 composed domain repositories (`RunLifecycle`, `Execution`, `DataFlow`, `Query`); recorder is now a pure delegation facade.
- *Plugins SDA restructure* — flat `plugins/` reorganised into `infrastructure/`, `sources/`, `transforms/`, `sinks/`; 247 files changed; about 200 imports rewritten.
- *Protocol relocation L3 to L0* — `SourceProtocol`, `TransformProtocol`, `SinkProtocol`, `BatchTransformProtocol`, `GateResult` moved from `plugins/infrastructure/` to `contracts/`; eliminates engine-to-plugins layer violation.
- *ADR-006*: Layer Dependency Remediation; full architecture analysis (23 documents covering all 13 subsystems); security posture brief for v0.3.0.

== RC-3.4 (10 Mar) — Systematic Hardening
- *191-bug triage*, mutation testing, code-quality sweep — no new features; pure correctness work.
- *PayloadNotFoundError* domain exception across `PayloadStore` / `Filesystem` / `Mock` (replaces generic `KeyError`).
- *Deep-freeze utilities*: `deep_freeze()`, `deep_thaw()` recursing into tuples, frozensets, `MappingProxyType`; `slots=True` on every frozen DC; central `freeze_fields()` API.
- *Construction-time validation* — `__post_init__` on 12 dataclass types, including coalesce checkpoints (non-empty IDs, non-negative timing, disjoint branch keys).
- *Exception hygiene* — `from None` replaced with `from exc` across 16 files; 22 broad `except Exception` catches narrowed; 6 silent-skip paths converted to explicit crashes.
- *`hasattr()` banned unconditionally* (3 occurrences replaced).
- *stdlib logging to structlog* across batch mixin, multi-query, Azure-blob source, Azure batch.
- Agentic-code threat-model discussion paper (v0.1--v0.4) with MkDocs wiki and LaTeX pipeline.

== Sized

- *475 commits across 26 calendar days* (25 active). One large multi-day burst around RC-3.3 cutover (29 commits 2026-03-01, 24 on 2026-03-05, 24 on 2026-03-08); calmest stretch of the project at 19 commits per active day.

= Period 5 — RC-4 Plugins and RC-5 Cut (11 March -- 3 April 2026)

#callout(kind: "success", title: "Outcome")[
  The first external-system plugins shipped (Dataverse, RAG); the
  first pipeline-level orchestration primitives shipped (`depends_on`,
  commencement gates); the *Web UX Composer* was cut as RC-5 — a
  chat-first, LLM-assisted pipeline authoring surface.
]

== RC-4.0 (mid-March) — Plugins, Contracts, Correctness
- *Dataverse source + sink* — OData v4 REST API via `DataverseClient`, structured OData and FetchXML, pagination, SSRF validation, rate limiting. 288 new tests.
- *RAG retrieval transform* — full RAG with lifecycle management; `RetrievalProvider` protocol with `ChromaSearchProvider` (ephemeral/persistent/client modes) and `AzureSearchProvider`; three query modes (`field`, `template`, `regex`); three context modes (`numbered`, `separated`, `raw`); `PluginRetryableError` base exception.
- *Output schema contract enforcement* — `_output_schema_config` class attribute; `FrameworkBugError` if a transform declares fields but omits the schema config.
- *Audit provenance boundary* — LLM audit fields stored in `success_reason["metadata"]` instead of polluting row data; `payload_store` removed from `PluginContext`.
- *Freeze / serialize coherence* — canonical-JSON now natively handles `MappingProxyType`, `tuple`, `frozenset` in `contracts/hashing.py`.
- *CI enforcement* — `enforce_freeze_guards.py` (AST-based) detecting bare `MappingProxyType` wraps and `isinstance` guard-skips in `__post_init__`.
- *WebScrape SSRF allowlist* — three-tier validation (always-blocked, user CIDR, standard blocked); 62 new tests.
- *errorworks migration* — ChaosLLM, ChaosWeb, ChaosEngine moved to external `errorworks` PyPI package.
- *RC4-Bugsweep*: 64 bugs across 13 clusters closed.

== RC-4.1 (late March -- 2 April) — RAG Ingestion Pipeline
- *ChromaSink* — write to ChromaDB with three `on_duplicate` modes (`overwrite` / `skip` / `error`); `FieldMappingConfig` (explicit, no defaults); ChromaDB metadata type validation at write time.
- *`depends_on`* — top-level config key for pre-run pipelines; `bootstrap_and_run()`; circular-dependency detection; sequential execution; `DependencyRunResult` audit DC.
- *Commencement gates* — go/no-go conditions evaluated after dependencies; uses `ExpressionParser` AST whitelist; deep-frozen `context_snapshot`; explicit `collection_probes` config.
- *Readiness contracts* — `check_readiness()` on `RetrievalProvider`; `CollectionReadinessResult`; `RetrievalNotReadyError`; `ChromaSearchProvider` and `AzureSearchProvider` implementations.
- *`preflight_results` audit table* — dependency runs and gate evaluations recorded per run.
- *End-to-end RAG example* — `examples/chroma_rag_indexed/` with indexing + query pipelines.
- *Tier 1 audit integrity hardening* — `require_int()` Tier 1 int validator (rejects Python's `bool`-is-`int` footgun); applied to 19 int fields in 13 audit DCs; TypedDict export records (15 typed shapes replacing `dict[str, Any]`); `CoalescePolicy` / `MergeStrategy` StrEnums; `Mapping[str, object]` write-path narrowing; `allow_nan=False` on 6 audit-path `json.dumps()` calls.

== RC-5 (cut 3 April) — Web UX Composer

A full web application platform for *chat-first pipeline composition*.
This is the surface that turned ELSPETH from a YAML-authored framework
into a system non-pipeline-engineers can drive.

#data-table(
  columns: 2,
  header: ([Subsystem], [What now works]),
  align-rules: (left + horizon, left + horizon),
  [`elspeth web` CLI],
  [FastAPI app factory with `[webui]` extra, `WebSettings` config, default port 8451, serves the React SPA],
  [Frontend],
  [Vite-built React SPA, `/api` and `/ws` proxy, *DTA / AGDS theming* (deep teal, green accent, GOLD), logout, session creation guards, archive sessions, confirm destructive actions, version loading, skip-to-content, reduced motion, touch-target sizing],
  [Auth],
  [`AuthProvider` protocol with `LocalAuthProvider` (bcrypt + JWT), `OIDCAuthProvider` (JWKS discovery), `EntraAuthProvider` (tenant + group claims); `get_current_user` FastAPI dependency; login / refresh / profile / config routes; configurable registration (`open` / `email_verified` / `closed`); python-jose to PyJWT migration],
  [Plugin catalog],
  [`CatalogService` protocol + REST routes],
  [Sessions],
  [SQLAlchemy Core tables + migrations; `SessionServiceImpl` with CRUD + versioning + run enforcement (`RunAlreadyActiveError`); fork-from-message; TOCTOU race elimination via DB-level constraints; thread-pool DB calls off the async loop; orphan cleanup in FastAPI lifespan],
  [Blob storage],
  [6 phases — data model, REST API, frontend integration, composer tools, execution integration, schema inference; upload dedup; quota enforcement; file cleanup],
  [Secret references],
  [`SecretResolution` audit extension for `env` and `user` sources; `resolve_secret_refs()` tree-walk for `$secret{name}` references; `ServerSecretStore` + `WebSecretService` with allowlist + fingerprint audit; REST + composer + execution + frontend wiring],
  [Execution],
  [`ExecutionServiceImpl` with WebSocket progress; cancel-vs-execute race closure; late-WebSocket client seeding],
  [Pipeline composer],
  [Frozen `SourceSpec` / `NodeSpec` / `EdgeSpec` / `OutputSpec` / `PipelineMetadata`; composition tools + YAML generator; `ComposerService` LLM tool-use loop; sub-4x hardening (dual-counter loop guard, discovery cache, partial state recovery, rate limiting, tool registry)],
  [Pipeline inspector],
  [Inspector UX overhaul, graph readability, version selector, catalog drawer],
  [Pipeline composer MCP server],
  [`elspeth-composer` MCP server — full toolset over Model Context Protocol; pipeline-composer skill pack; wave-4 tools (`clear_source`, `explain_validation_error`, `list_models`, `preview_pipeline`)],
  [Sink failsink pattern],
  [`RowDiversion`, `SinkWriteResult`, `DIVERTED` outcome, `rows_diverted` counter, `on_write_failure` mandatory config field, `BaseSink._divert_row()`, automatic `__failsink__` DIVERT edges in DAG builder],
  [DAG schema propagation],
  [`output_schema_config` as single source of truth across source / transform / gate / aggregation / coalesce],
  [Frontend UX refresh (A1--A7)],
  [Categorized blob folders; Markdown + Mermaid rendering (with DOMPurify); route validation through chat; 50/50 panel split; secrets in toolbar; per-node validation indicators; three-state pipeline status indicator],
  [Guard symmetry scanner],
  [`enforce_guard_symmetry` CI tool — every Landscape write site must have a corresponding read guard],
  [`TokenRef` type],
  [Bundled `token_id + run_id` frozen DC; `AuditIntegrityError` loader guards; `coalesce_tokens(TokenRef)`; `_validate_token_run_ownership(TokenRef)`],
  [Exception hygiene],
  [`TIER_1_ERRORS` canonical tuple applied across all layers],
  [Bug closure campaign],
  [About 100+ P1 bugs closed plus a post-cut systematic sweep of about 130 additional bugs],
  [Test hygiene],
  [About 500 low-value tests deleted, about 200 gap-filling tests added (net: better coverage of actual behaviour)],
)

= Period 6 — RC-5.1 Composer Correctness (4 April -- 11 May 2026)

#callout(kind: "success", title: "Outcome")[
  RC-5 had shipped the surface; RC-5.1 made it *answer accurately*.
  The composer's authoring loop, validation surface, and run-evidence
  views all received targeted hardening; the run-evidence story was
  extended with discard summaries, failure-sample aggregation, and a
  cancellation-requested badge.
]

#callout(kind: "advisory", title: "Notable incident — wire-visible-fabrication defect, contained")[
  The most consequential fix in this period closed a *Tier-1
  wire-visible-fabrication defect*: during a development-environment
  evaluation run, the composer's LLM had copied a placeholder example
  value (`compliance@example.com`, supplied in a skill-prompt code
  snippet) into the `web_scrape.http.abuse_contact` HTTP header on
  requests to three Australian Government websites. The defect was
  detected by ELSPETH's own audit trail (the evaluation harness, not
  production traffic), and the audit record is the reason this
  incident is documented here. The remediation —
  `<OPERATOR_REQUIRED>` sentinels for identity-bearing fields, a hard
  rule against silent operator-input rewrites, and an implicit-
  decision disclosure block — is described under #emph[Validator and
  Pipeline Authoring (RC-5.1)] below. *No production deployment was
  affected.*
]

== Surface expansion

- *Substrate-first README* — ELSPETH framed as a high-assurance pipeline substrate with two authoring surfaces (hand-edited YAML for operators, Web Composer for LLM-assisted authoring by non-pipeline-engineers); validator-mediated authoring framing; expanded composer surface description; audited composer tool loop; runtime-shaped validation and preflight; run-evidence endpoints; cancellation visibility.
- *Composer reliability + operator visibility* — deterministic composer calls (temperature/seed in audit sidecar); prompt-cache-aware audit; reasoning metadata capture; *advisor escalation contract* (frontier-model escalation gated behind mechanically validated trigger categories); hard-mode evaluation harness; advisor-conditional skill markers (`<!-- ADVISOR-ONLY -->` / `<!-- ADVISOR-DISABLED -->`).
- *Plugin / contract surface* — plugins can publish semantic facts, requirements, comparison outcomes, and composer assistance text; *10 statistical batch plugins* added with runnable examples (`batch_distribution_profile`, `batch_experiment_compare`, `batch_classifier_metrics`, `batch_paired_preference`, `batch_drift_compare`, `batch_outlier_annotator`, `batch_data_quality_report`, `batch_top_k`, `batch_threshold_summary`, `batch_effect_size`); richer missing-dependency / schema vocabulary / repair hints.
- *Two-axis terminal outcome model* — lifecycle outcome separated from terminal path/provenance; run accounting split into routed-success/failure, token-lifecycle counts, source-row counts, discard summaries, closure integrity.
- *CI / policy / test surface* — gates added or tightened for component types, guard symmetry, audit-evidence nominal typing, Tier-1 decoration, contract manifests, composer exception channels, composer catch order, tier-model allowlists; *Frontend Playwright baseline* booting FastAPI + Vite for E2E.

== RC-5.1 specifics (composer correctness)

- *`identity_node_advisory` validator* — `validate_pipeline` detects identity passthrough nodes silently degrading observed-sink lineage; gated by an exemption matrix.
- *Composer pipeline recipes* — `apply_pipeline_recipe` MCP tool + two initial templates; deep-frozen `RecipeSpec.slots`; recipes emit schema-valid `llm` and `type_coerce` options.
- *Source inspection MCP tool* — `inspect_source` surfaces external-data shape and silent-failure modes (e.g. all rows quarantined) as warnings.
- *Forced-repair loop with proof diagnostics* — `preview_pipeline` runs a proof step; `compute_proof_diagnostics` verifies blob `content_hash` (stale blob can no longer pass the proof gate); `_BLOCKING_DIAGNOSTIC_CODES` is the structural source of truth.
- *Audit-backend skill + recipe-first fork-coalesce shape* in composer skill pack; mandatory advisor-escalation gate for Recipe \#10 (fork+coalesce).
- *Convergence-suite scenarios* — URL-text smoke, mocked-LLM integration, end-to-end forced-repair, end-to-end `apply_pipeline_recipe`, fork-and-coalesce regression.
- *Composer authoring affordances* — "Use in pipeline" prefills chat input; chat code blocks gain syntax highlight + copy-to-clipboard; `secret_ref` inline form; resize-handle keyboard arrows; touch-friendly hit zone.
- *`<OPERATOR_REQUIRED>` sentinels* — replace literal example values for identity-bearing fields (`web_scrape.http.abuse_contact`, `scraping_reason`); explicit resolution order (operator-supplied to deployment-identity to ask before `set_pipeline`).
- *Hard rule against silent operator-input rewrites* — any normalisation must be confirmed or routed through a recorded step that appears in YAML.
- *Implicit-decision disclosure* ("Decisions I made on your behalf") — Build Summary now enumerates operator-invisible authoring decisions with provenance markers (`default` / `picked` / `deployment-identity` / `operator-supplied`).
- *Run-evidence widgets* — `RunOutputsPanel` (full audit-evidence manifest, downloadable artifacts gated by `downloadable` flag); cancellation-requested badge separate from terminal `cancelled`; GraphView viewport preservation across topology changes; `data_dir` resolved to absolute path at validation time; source-inspection silent-failure surfacing; failure-sample aggregation in run-level errors.
- *Audit-integrity test coverage* — direct unit coverage added for *ADR-019 deferred-invariant sweep*, `_validate_token_row_ownership`, `link_validation_error_to_row` branches, all 12 entries in `_REQUIRED_COMPOSITE_FOREIGN_KEYS`, and SSRF blocked-IP residual coverage (including `::ffff:0:0/96` and seven other previously untested boundary cases).
- *Tier-1 panel-review accessibility pass* — `aria-controls` IDREF when run collapsed; `aria-expanded` on `RunsView` Inspect; health-check banners downgraded to `role=status`; nested `aria-live` removed from `ComposingIndicator`; light-theme `--color-status-empty` override.

= Period 7 — RC-5.2 Guided Composer and Recovery (12--19 May 2026)

#callout(kind: "success", title: "Outcome")[
  The Web Composer crossed from "best-effort interactive surface" into
  "audited, recoverable authoring system". Every model call, every
  tool dispatch, every redacted tool payload, every persisted
  transcript row, every recovery diff, and every operator-visible
  failure cause now shares one evidence story.
]

== Composer Guided Mode

- *Guided wizard* — structured-protocol pipeline authoring for first-time users; source -> sink -> transforms in three steps; closed six-turn taxonomy; deterministic recipe pre-match; LLM is read-only with respect to pipeline state. Ships alongside the unmodified freeform composer; mode transition uses progressive disclosure.
- *`ComposerLLMCall` audit channel* — every `solve_chain` invocation records a row (provider, model, status, latency, prompt/completion tokens). Pairs with the existing `ComposerToolInvocation` audit channel.

== Composer Progress Persistence (Phases 1A to 4)

- *Phase 1A — schema* — new `chat_messages` audit columns (`tool_call_id`, `sequence_no`, `writer_principal`, `parent_assistant_id`) with biconditional CHECK constraints pinning OpenAI-shaped tool-call linkage; `composition_states.provenance` enum (`tool_call` / `convergence_persist` / `plugin_crash_persist` / `preflight_persist` / `session_seed` / `session_fork`); `run_events` table; `audit_access_log` table (INERT for Phase 3+); per-session indices.
- *Phase 1B — single-transaction primitive* — `SessionServiceImpl.persist_compose_turn` (sync + async); `StatePayload` / `_ToolOutcome` / `RedactedToolRow` / `AuditOutcome` DTOs; advisory-lock primitive (`pg_advisory_xact_lock` on Postgres, per-session `RLock` on SQLite); sequence-number reservation under session write lock.
- *Phase 1C — Postgres portability* — `@pytest.mark.testcontainer` lane spinning up ephemeral Postgres per test; exercises `pg_advisory_xact_lock`, commit-wins concurrency, and Postgres-specific blob `ready_hash` partial uniqueness.
- *Phase 2 — redaction walker + MANIFEST* — `web/composer/redaction.py` grew from a 42-line stub to a *2,752-line walker*; 38-entry MANIFEST (10 type-driven + 28 declarative) with Pydantic argument models; Pydantic-first ARG_ERROR routing (`ToolArgumentError` re-raise; LLM-facing message names argument-bundle only — rev-2 BLOCKER_A leak discipline); structured Pydantic detail preserved on `__cause__` via `canonicalize_pydantic_cause`; adequacy guard pinning manifest-registry parity and a byte-identical redaction snapshot; F1--F6 hardening (completeness Hypothesis property tests, walker-guard parity, summarizer-contract Hypothesis, label-gate CI workflow, drift guards).
- *Phase 3 — compose-loop persistence* — `_compose_loop` persists assistant messages, tool-call breadcrumbs, redacted tool payloads, and composition-state snapshots through `persist_compose_turn`; audit-first contract preserved through tool failure, mid-turn cancellation, and plugin-crash recovery; per-turn tool-call cap with `tool_call_cap_exceeded` reason code; `include_tool_rows=true` opt-in for audit-grade transcript access recorded through `audit_access_log`; property + integration tests for audit counter conservation, manifest redaction, cancellation commit windows, failed-turn tool-response counts, no-op behaviour.
- *Phase 4 — frontend recovery* — recovery panel detecting recoverable composer failures; assistant transcript + redacted tool rows + before/after state diff rendered via `RecoveryTranscript` / `RecoveryDiff` / `RecoveryPanel`; frontend `npm run lint` gate added.

== RC-5.2 Hotfix Integration (folded back to `main`)

- *Auth + audit hardening* — local/Entra audit token issuance, auth-failure classes, local login outcomes, refresh-provider invariants, provider outages, web-run attribution to Landscape; JWKS-failure detail redacted; token-response caching suppressed.
- *Execution + validation hardening* — web execution classifies validation errors, sanitises broad execution errors, persists resolved run config, rejects misplaced secret refs, preserves guided audit persistence failures.
- *Engine / plugin correctness fixes* — checkpoint resume parsing; empty coalesce checkpoint state; pending batch row identities; JSON sink parent creation; sink preflight collision timing; Web Scrape fail-closed boundaries; LLM provider preflight; shared LLM telemetry helpers.
- *Frontend accessibility + theming fixes* — improved contrast on guided/catalog/run UI; forced-colors fallback; theme initialisation + cross-tab sync; screen-reader-safe status symbols; catalog retry controls; keyboard shortcut support; preserved plugin descriptions.

== Phase 6 / 7 / 8 (post-5.2 merge stream)

- *Phase 5* — chat-data-entry: dynamic 1-row source from chat text for hello-world / canonical-test-case shapes.
- *Phase 6* — completion gestures: Save-for-review / Run-analysis / Execute / Copy-YAML differentiated by persona.
- *Phase 7* — catalog reshape (16 / 16a / 16b / 16c batched): drawer reframed from interactive toolkit to searchable reference; plugin-coverage gates calibrated for `report_assemble`; `i1`/`i2`/`i3` fixes (drawer error log, snapshot lock, NETWORK retirement).
- *Phase 8* — final polish sweep: dead-code removal, lint, CSS tokens; four contrast-scan surfaces enumerated for Phase 9; CICD allowlist burn-down merged.

#callout(kind: "advisory", title: "Operational change of note")[
  Sessions DB schema deployment requires recreation. Phase 1A columns,
  tables, CHECKs, and partial unique indices are #strong[not applied
  via Alembic]. Operator stops the service, archives the old
  `sessions.db`, and restarts; the bootstrap creates the new schema on
  first start. Procedure documented in
  `docs/runbooks/staging-session-db-recreation.md`.
]

= Three-tier trust model

// Native-Typst diagram (cetz) — see theme.typ for the migration
// rationale away from the previous Mermaid PNG embed.
#figure(
  pdf.artifact(align(center, diagram-trust-tiers())),
  caption: [Three-tier trust model — the contract at each boundary.
    Tier 3 (External, zero trust): source plugins (CSV, JSON,
    Dataverse, Blob, Web); validate, coerce, quarantine, record
    absence as None. Tier 2 (Pipeline, elevated trust): transforms,
    gates, aggregations on type-safe values; no coercion — operations
    are wrapped instead. Tier 1 (Our data, full trust): Landscape
    audit DB, checkpoint state, hashes; crashes on any anomaly and
    stays pristine. The boundary into Tier 1 reads straight and writes
    atomically; the read-guard back-arrow (TIER_1_ERRORS) enforces
    the integrity contract on every read out of Tier 1.],
)

= Capability accretion

A timeline of which capability arrived in which RC. Capabilities are
grouped by feature surface; the column shows the first release in
which the capability was present in shipping form.

#data-table(
  columns: 2,
  header: ([Capability], [First shipped]),
  align-rules: (left + horizon, left + horizon),
  ..d.capability-accretion.map(((cap, rc)) => ([#cap], [#rc])),
)

= Cumulative output snapshot

== Code

#data-table(
  columns: 6,
  header: ([Measure], [RC-1], [RC-2 (0.1.0)], [RC-3.4], [RC-5.0], [Current (RC-5.2 + Phase 8)]),
  align-rules: (left + horizon, right + horizon, right + horizon,
    right + horizon, right + horizon, right + horizon),
  [Commits to date], [782], [1,593], [2,595], [3,073], [4,521],
  [Framework test suite],
  [mutation + unit + integration (Phase 1--3 regime)],
  [8,138 collected (v2)], [\~10,500],
  [\~10,500 + composer / web], [\~10,500 framework],
  [Composer-track tests (RC-5.2)], [--], [--], [--], [--],
  [3,159 backend + 447 frontend],
  [Major plugin surface], [13+ plugins], [adds ChaosLLM],
  [adds LLM consolidation], [adds Dataverse, RAG, ChromaSink],
  [adds 10 statistical batch plugins, recipes],
)

== Process / discipline first-enforcement

#data-table(
  columns: 2,
  header: ([Discipline], [First enforced]),
  align-rules: (left + horizon, left + horizon),
  [Canonical JSON via RFC 8785 + SHA-256], [RC-1],
  [Three-tier trust model (Tier 1 / Tier 2 / Tier 3)], [RC-1],
  [`frozen=True` + `slots=True` on audit DTOs], [RC-3.2 / 3.3],
  [`enforce_tier_model.py` CI gate], [RC-1 Hardening],
  [`enforce_freeze_guards.py` CI gate], [RC-4.0],
  [`enforce_guard_symmetry` CI gate], [RC-5.0],
  [Layer-import enforcement (L0 to L1 to L2 to L3)], [RC-3.3],
  [Test factory architecture (`make_context`, `make_recorder_with_run`)], [RC-3.3],
  [4-layer model + ADR-006], [RC-3.3],
  [Property-based testing (Hypothesis) across SSRF, ChaosLLM, DAG, triggers, routing, schema contracts, reorder buffer, orchestrator lifecycle], [RC-2.4 onwards],
  [Mutation-survivor regimen (> 71 new tests killed in RC-4.0 alone)], [RC-3.3 onwards],
  [`hasattr()` banned unconditionally], [RC-3.4],
  [`TIER_1_ERRORS` canonical exception tuple], [RC-5.0],
  [Composer audit-MANIFEST adequacy guard + byte-identical snapshot], [RC-5.2],
  [CICD-allowlist-audit skill (periodic 4--8 week pass)], [RC-5.2],
)

= Post-RC-5.2 follow-up themes

The following themes are visible from the release evidence and remain useful
planning inputs after RC-5.2:

- *Composer correctness cluster* — validator parity, runtime dry-run, operator visibility (`elspeth-528bde62bb`).
- *Fork / coalesce audit-integrity epic* — schema reconciliation, field provenance, merge safety (`elspeth-e20903300c`).
- *Web auth hardening* (OIDC / Entra / JWKS) (`elspeth-250f698aaf`).
- *Web sessions + Alembic-env reconciliation* (`elspeth-ef52049338`).
- *Plugin Expansion Phase 1* — web research pipeline (OpenSearch, browser scrape, report sink, Chroma upgrade) (`elspeth-868c55d712`).
- *Composer progress persistence* — tool-call breadcrumbs and partial drafts surviving long-running failures (`elspeth-90b4542b63`).

= Sources

- `docs-archive/2026-05-19-docs-cleanout/CHANGELOG-RC1.md` — Pre-RC1 and RC-1 Hardening (Jan 12 -- Feb 2, 2026) — archived snapshot
- `docs-archive/2026-05-19-docs-cleanout/CHANGELOG-RC2.md` — RC-2.0 through RC-2.5 (Feb 2--12, 2026) — archived snapshot
- `/CHANGELOG.md` — RC-3.2 through RC-5.2 + Unreleased (Feb 13, 2026 -- present)
- `docs-archive/2026-05-19-docs-cleanout/docs/release/feature-inventory.md` — RC-3.3 feature reconciliation (1 March 2026) — archived snapshot
- `docs-archive/2026-05-19-docs-cleanout/docs/release/rc4-executive-brief.md` — RC-4 work-package summary (3 March 2026) — archived planning brief
- Git history snapshots for RC-1, RC-2, RC-5.2, and `main`
- Maintainer planning snapshot (19 May 2026)
