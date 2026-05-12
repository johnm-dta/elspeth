# Changelog

All notable changes to ELSPETH are documented here.

---

## [Unreleased]

### Added

- **Composer guided mode** — new structured-protocol wizard for first-time
  pipeline authors. Source → sink → transforms in three steps; closed
  six-turn taxonomy; deterministic recipe pre-match; LLM-read-only with
  respect to pipeline state. Ships alongside the unmodified freeform
  composer; mode transition uses progressive disclosure.
  See `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md`.

---

## [0.5.1] - 2026-05-11 (RC-5.1 — Composer Correctness, Validator Hardening, and Audit-Integrity Coverage)

RC-5.1 is a correctness and assurance follow-up to RC-5. The Web Composer's
authoring loop, validation surface, and run-evidence views all received targeted
hardening; a new `identity_node_advisory` validator detects identity passthroughs
that obscure observed-sink lineage; the `data_dir` config is now resolved to an
absolute path at validation time; pipeline recipes, source inspection, and a
forced-repair loop with proof diagnostics extend the composer authoring surface;
and the Landscape audit-integrity test surface gains direct coverage for four
ADR-019-family invariants that previously had zero unit tests. Frontend UX received
a Tier-1 panel-review pass covering accessibility, focus management, and operator
visibility.

### Added

#### RC5-UX Substrate, Composer, and Execution Evidence

- **Substrate-first README story** — README now presents ELSPETH as a
  high-assurance pipeline substrate with two authoring surfaces: hand-edited
  YAML for operators and the Web Composer for LLM-assisted authoring by
  non-pipeline-engineers.
- **Validator-mediated authoring framing** — the README now emphasizes that
  both surfaces target the same primitives, runtime assembly, graph-validation
  contracts, executor, Landscape audit trail, and run-accounting model instead
  of treating the web UI as a bolt-on workflow builder.
- **Expanded Web Composer surface** — authenticated sessions, versioned
  composition state, blob management, secret references, chat-first pipeline
  authoring, graph/spec/YAML inspection, validation, execution, cancellation,
  diagnostics, and output artifact review are now described as part of the
  authoring surface.
- **Audited composer tool loop** — the composer surface is now described as a
  tool-governed authoring loop: plugin discovery, state mutation, validation,
  YAML export, blob tools, secret-reference tools, and optional advisor hints
  all happen through explicit tool contracts.
- **Runtime-shaped validation and preflight** — composer previews, YAML export,
  `/validate`, and `/execute` are documented as using runtime assembly and graph
  validation contracts instead of a separate best-effort UI validator, while the
  compiled-artifact compiler boundary is stated as future direction.
- **Run evidence endpoints** — web execution now exposes Landscape-derived run
  accounting, diagnostics snapshots, discard summaries, and the full output
  artifact manifest/content surfaces for a run.
- **Cancellation visibility** — in-progress runs can carry a distinct
  cancellation-requested state while active work drains toward a terminal
  `cancelled` status.

#### Composer Reliability and Operator Visibility

- **Deterministic composer calls** — composer LLM requests use deterministic
  sampling where supported and record temperature/seed metadata in the LLM
  audit sidecar.
- **Prompt-cache-aware composer audit** — provider cache counters and
  Anthropic-style prompt-cache markers are captured without fabricating missing
  values.
- **Reasoning metadata capture** — composer LLM audit records can carry
  provider-reported reasoning token counts and reasoning artifacts when a
  provider exposes them; normal chat history still hides those internals.
- **Advisor escalation contract** — the optional advisor tool is gated behind
  explicit trigger categories so the composer can ask for frontier-model help
  only under mechanically validated conditions.
- **Hard-mode composer evaluation harness** — reusable shell tooling and
  scenario fan-out capture validation transport failures, composer regressions,
  and per-row output evidence for demo-readiness checks.
- **Advisor-conditional skill markers** — composer prompt assembly now strips
  both `<!-- ADVISOR-ONLY -->` and `<!-- ADVISOR-DISABLED -->` regions from the
  skill markdown depending on whether the advisor tool is enabled, so an
  advisor-disabled deployment can no longer leak `request_advisor_hint`
  guidance (anti-fabrication rule, fork+coalesce table row, Recipe #10
  escalation, §10b read-gate) to the composer LLM.

#### Plugin and Contract Surface

- **Plugin-declared semantic contracts** — plugins can now publish semantic
  facts, requirements, comparison outcomes, and composer assistance text; the
  composer, MCP server, HTTP validation, and `/execute` surfaces expose the
  same contract shape.
- **Statistical batch plugin family** — added runnable local examples for
  `batch_distribution_profile`, `batch_experiment_compare`,
  `batch_classifier_metrics`, `batch_paired_preference`,
  `batch_drift_compare`, `batch_outlier_annotator`,
  `batch_data_quality_report`, `batch_top_k`, `batch_threshold_summary`, and
  `batch_effect_size`.
- **Value-source and schema-contract guidance** — plugin catalogs and composer
  guidance now carry richer missing-dependency, schema vocabulary, and repair
  hints so an invalid pipeline points at the contract it violated.

#### Audit and Accounting Model

- **Two-axis terminal outcome model** — RC5-UX separates lifecycle outcome from
  terminal path/provenance so success, failure, routed, discarded, fallback, and
  structural bookkeeping cases no longer overload a single row outcome.
- **Run accounting split** — routed success/failure, token lifecycle counts,
  source-row counts, discard summaries, and closure integrity are reported as
  distinct units instead of being inferred from one `rows_routed` number.
- **Per-tool-call composer audit** — both the web composer and the standalone
  composer MCP server persist the tool-call decision trail that produced a
  pipeline.

#### CI, Policy, and Test Surface

- **Policy gates expanded** — branch work added or tightened gates for
  component types, guard symmetry, audit evidence nominal typing, tier-1
  decoration, contract manifests, composer exception channels, composer catch
  order, and tier-model allowlists.
- **Frontend Playwright baseline** — a real browser E2E harness now boots the
  FastAPI backend and Vite frontend, with an initial smoke proof and fixme
  specs for composer-correctness acceptance paths.

#### Validator and Pipeline Authoring (RC-5.1)

- **`identity_node_advisory` validator check** — `validate_pipeline` now detects
  identity passthrough nodes wired between transforms and observed sinks, where
  the passthrough silently degrades observed-sink lineage. The check is gated
  by an exemption matrix locked in by tests, surfaces as an actionable repair
  hint in the composer, and is enforced on the validation happy path.
- **Composer pipeline recipes** — `apply_pipeline_recipe` MCP tool plus two
  initial templates compose a multi-node pipeline from a single named recipe,
  with deep-frozen `RecipeSpec.slots`, `SlotSpec.default` validated against
  `slot_type` at construction, recipes that emit schema-valid `llm` and
  `type_coerce` options, and recipe `blob_id` placed at source top-level so
  `set_pipeline` can resolve it.
- **Source inspection MCP tool** — new `source_inspection` module and
  `inspect_source` MCP tool surface external-data shape and silent-failure modes
  as warnings, supporting "look before you wire" composer sessions; hostile-input
  coverage included for `inspect_blob_content`.
- **Forced-repair loop with proof diagnostics** — `preview_pipeline` runs a
  proof step that produces a `proof_diagnostics` array; the composer's
  forced-repair loop is wired to those diagnostics, fires on the resumed-session
  first turn, and plumbs `repair_turns_used` into `composition_states.composer_meta`.
  `compute_proof_diagnostics` verifies blob `content_hash` so a stale blob can
  no longer pass the proof gate. `_BLOCKING_DIAGNOSTIC_CODES` is now the
  structural source of truth for blocking diagnostics.
- **Audit-backend skill + recipe-first fork-coalesce shape** — composer skill
  pack adds an audit-backend skill and reorganises the fork-coalesce guidance
  to be recipe-first; canonical fork pattern aligned to the
  `validate_boolean_routes` contract; mandatory advisor escalation gate for
  Recipe #10 (fork+coalesce) shapes.
- **Convergence-suite test scenarios** — new evaluation scenarios cover
  end-to-end convergence (URL-text smoke, mocked-LLM integration, end-to-end
  forced-repair through the real `_compose_loop`, end-to-end
  `apply_pipeline_recipe` through the proof step) plus a pure scoring function;
  fork-and-coalesce regression scenario locks in skill commit `a2d9706b`.
- **Composer authoring affordances** — `Use in pipeline` action in the plugin
  catalog prefills the chat input; chat code blocks gain syntax highlighting
  and a copy-to-clipboard control; `secret_ref` advertises an inline form for
  new-node credentials; resize-handle keyboard arrows align with value
  direction with a static affordance and touch-friendly hit zone.
- **`<OPERATOR_REQUIRED>` sentinels for identity-bearing fields** — the
  composer skill replaces literal example values for
  `web_scrape.http.abuse_contact` and `scraping_reason` with angle-bracket
  sentinels and an explicit resolution order (operator-supplied →
  deployment-identity → ask the operator before `set_pipeline`). The
  angle-bracket form is intentional: a placeholder validator (tracked
  separately) can mechanically reject any YAML that still carries a
  sentinel, providing a structural safety net for the prompt-level rule.
- **Hard rule against silent operator-input rewrites** — composer skill now
  forbids silent normalisations of operator-supplied strings (e.g. prepending
  `https://` to a bare hostname, lowercasing, trailing-slash strip). Any
  rewrite must either be confirmed by the operator or routed through a
  recorded normalisation step (`value_transform` etc.) so it appears in the
  YAML and the build summary.
- **Implicit-decision disclosure ("Decisions I made on your behalf")** — new
  Build Summary Discipline subsection requires the composer to enumerate
  operator-invisible authoring decisions (identity headers, model/provider/
  temperature, output shape and routing, format choices, allowlist defaults,
  surviving operator-input rewrites) with explicit provenance markers
  (`default` / `picked` / `deployment-identity` / `operator-supplied`).

#### Run Evidence and Operator Visibility (RC-5.1)

- **Run outputs panel** — frontend `RunOutputsPanel` exposes the full
  audit-evidence manifest for a run, with downloadable artifacts gated by a
  per-artifact `downloadable` flag; backed by a new `/artifacts/preview`
  execution endpoint.
- **Cancellation-requested badge** — runs whose cancellation has been requested
  but not yet drained now carry a distinct badge style, separate from the
  terminal `cancelled` state.
- **GraphView viewport preservation** — composer GraphView preserves the
  operator's pan/zoom across topology changes, so iterative edits no longer
  reset the view to a default `fitView`.
- **`data_dir` resolved to absolute path** — `WebSettings` now resolves
  `data_dir` to an absolute path at validation time, eliminating a class of
  ambiguity where relative paths were interpreted against different working
  directories at validate vs. run time.
- **Composer source-inspection silent-failure surfacing** — Tier-3 source
  inspection surfaces silent-failure modes (e.g. all-rows quarantined) as
  warnings rather than treating success-with-zero-results as a quiet pass.
- **Failure-sample aggregation in run-level errors** — new
  `web/execution/failure_samples` module aggregates the top distinct
  `transform_errors` rows so a failed run's top-level error message carries
  actionable detail rather than a single bubble-up exception string.

#### Audit Integrity Test Coverage (RC-5.1)

- **ADR-019 deferred-invariant sweep** — direct unit coverage for the
  `sweep_deferred_invariants_or_crash` run-end invariant enforcer (was
  previously zero unit tests).
- **DataFlowRepository `_validate_token_row_ownership`** — direct coverage for
  the cross-row lineage corruption guard (previously never directly tested).
- **`link_validation_error_to_row` branches** — direct coverage for quarantine
  lineage exactness across all branches.
- **`_REQUIRED_COMPOSITE_FOREIGN_KEYS`** — exhaustive coverage for all 12
  composite foreign-key entries (previously 11 of 12 untested).
- **SSRF blocked-IP coverage** — closes residual blocklist coverage including
  the `::ffff:0:0/96` IPv4-mapped-IPv6 range and seven other previously
  untested boundary cases in `web_scrape`.

#### Contracts and Lifecycle (RC-5.1)

- **Symmetric `_on_start_called` / `_on_complete_called` lifecycle flags** —
  runtime-config protocols now expose paired start/complete lifecycle flags so
  asymmetric implementations (start without complete, or complete without
  start) become structurally detectable.
- **Autouse catalog fixture + derived `RUNTIME_CONFIGS`** — contract tests now
  derive the runtime-config inventory from runtime introspection rather than
  a hand-maintained list, eliminating drift between the inventory and the
  actually-registered configs.

### Changed

- **README front door refreshed for RC-5** — the README now leads with the
  high-assurance substrate, the gap ELSPETH closes, the two authoring audiences,
  architecture at a glance, parallel YAML/Web Composer start paths, capability
  map, audit/assurance model, and shipped-vs-direction compiler status.
- **Web quickstart port corrected** — the README quickstart now uses the
  `elspeth web` default port `8451`.
- **Batch-specific LLM transforms retired from the public story** —
  `azure_batch_llm` and `openrouter_batch_llm` are no longer advertised. Use
  the regular `llm` transform with provider pooling/multi-query for LLM
  throughput and the statistical batch transforms for audit-attributable local
  aggregation.
- **Default `on_validation_failure` is now `discard`** — the default
  per-source validation-failure behaviour changed to `discard` with documented
  quarantine semantics, replacing the prior implicit fall-through.
- **Unknown-plugin composer error is now actionable** —
  `_prevalidate_plugin_options` surfaces an unknown plugin id as a structured,
  actionable rejection instead of a silent fail-open.
- **Composer rejected-mutation entries lead `validation.errors`** — when a
  composer mutation is rejected, the rejection entry is now the first item in
  `validation.errors`, so downstream UI surfaces and skill prose treat it as
  the primary diagnostic.
- **Augment-shape preflight failures preserve model prose** — when an
  augment-shape preflight fails, the composer now retains the model's original
  prose explanation instead of replacing it with a generic failure summary.
- **`apply_pipeline_recipe` surfaces destructive replacement** — applying a
  recipe over an existing pipeline now explicitly reports the replacement as
  destructive in the composer audit trail.

### Fixed

- **Wire-visible identity fabrication (Tier-1 audit-integrity defect)** — the
  composer skill previously carried `compliance@example.com` and a generic
  scraping reason as the canonical worked example for the
  `web_scrape.http.abuse_contact` / `scraping_reason` fields, and the LLM was
  copying those literals into generated YAML verbatim. A reproduced eval
  session shipped a fabricated abuse-contact email as an HTTP header to three
  external `.gov.au` sites — a confident wrong answer to a third party we have
  no relationship with. Closed by the `<OPERATOR_REQUIRED>` sentinel rewrite,
  the silent-rewrite hard rule, and the implicit-decision disclosure block
  (see *Validator and Pipeline Authoring (RC-5.1)*).
- **Composer skill correctness** — multi-commit sweep closing skill-text
  fabrication and silent-shape-downgrade loopholes, widening the grounding
  detector and adding `confirm`/`confirmed` to the grounding verb list, scoping
  state-claim grounding correction (Path 3), forbidding identity nodes between
  transforms and observed sinks, and narrowing the `ComposerResult` pairing
  invariant.
- **Composer accessibility (Tier-1 panel review)** — `aria-controls` IDREF
  remains resolvable when a run is collapsed; `aria-expanded` added to the
  `RunsView` Inspect button (now using the shared `.btn` class); health-check
  banners downgraded to `role=status`; nested `aria-live` removed from
  `ComposingIndicator`; `aria-live` scoped to the validation banner instead of
  the tab panel; light-theme `--color-status-empty` override added; validation
  indicator decoupled from the warning colour.
- **Composer SecretsPanel form recovery** — the form now recovers cleanly when
  `createSecret` fails, instead of leaving the panel in a wedged state.
- **Composer audit-shape symmetry** — `augment` and `replace` audit shapes now
  use symmetric producer-side invariants.

---

## [0.5.0] (RC-5 — Web UX Composer + Systematic Hardening)

Full web application platform for chat-first pipeline composition, three-provider authentication, session management with versioning, blob storage, secret management, background pipeline execution with WebSocket progress, and a React frontend themed to DTA/AGDS guidelines. Also: sink failsink pattern for per-row write failure routing, pipeline composer MCP server, DAG schema propagation (`output_schema_config` as single source of truth), declaration-trust / compiler-boundary hardening for pre-data runtime guarantees, frontend UX refresh (A1-A7), composer agent tooling (B1-B5) and skill pack update (C1-D4), guard symmetry CI scanner, `TokenRef` type, exception hygiene with `TIER_1_ERRORS`, a 200+ bug closure campaign across all subsystems, and a comprehensive test hygiene sweep removing ~500 low-value tests while adding ~200 gap-filling tests.

### Added

#### Declaration-Trust / Compiler-Boundary Hardening

- **Declaration-trust framework** — generalized ELSPETH's config-to-execution contract into a first-class declaration-trust system. Declarations trusted during graph construction and web validation — including pass-through behavior, declared input/output fields, schema mode, source guaranteed fields, sink required fields, and empty-emission governance — are now explicit, validated at compile time where possible, enforced by CI manifest/scanner guardrails, recorded in the run header via the runtime VAL manifest, and re-verified at runtime with audit-complete violation reporting. This moves the web UX closer to a true compiler front-end: a configuration that validates is one that satisfies every pre-data runtime guarantee ELSPETH can assess before Tier 3 data arrives.

#### Web UX Composer Platform

- **`elspeth web` CLI command** — FastAPI app factory with `[webui]` extra, `WebSettings` config model, and default port 8451. Serves the React SPA from `src/elspeth/web/frontend/dist/`.
- **React frontend bundle** — Vite-built SPA with `/api` and `/ws` proxying for development.
- **DTA/AGDS theming** — deep teal, green accent, and GOLD semantic colours matching Australian Government Design System guidelines.
- **Frontend UX** — logout UI, session creation guards, archive sessions, confirm destructive actions, version loading, bumped font sizes.
- **Accessibility** — skip-to-content links, reduced motion support, touch target sizing.

#### Authentication Subsystem

- **`AuthProvider` protocol** — pluggable identity model with `AuthenticationError` base exception.
- **`LocalAuthProvider`** — bcrypt password hashing with JWT token issuance.
- **`OIDCAuthProvider`** — OpenID Connect with JWKS discovery and key caching.
- **`EntraAuthProvider`** — Microsoft Entra ID with tenant validation and group claims.
- **`get_current_user` middleware** — FastAPI dependency for route-level authentication.
- **Auth routes** — login, token refresh, user profile, configuration endpoints.
- **Registration endpoint** — configurable mode (`open`, `email_verified`, `closed`).
- **python-jose → PyJWT migration** — replaced unmaintained library across all auth code.

#### Plugin Catalog

- **`CatalogService` protocol and implementation** — plugin discovery service with REST API routes wired into the app factory.

#### Session Management

- **SQLAlchemy Core table definitions** — session database schema with migrations.
- **`SessionServiceProtocol`** and `SessionServiceImpl` — CRUD, versioning, run enforcement, with `RunAlreadyActiveError`.
- **Session API routes** — full REST API with pagination, state pruning, upload hardening.
- **Fork-from-message** — create new session versions branching from specific conversation messages, with text source plugin.
- **TOCTOU race elimination** — DB-level constraints replacing application-level checks (batch 6).
- **Thread pool executor** — all DB calls moved off the async event loop (batch 5).
- **Orphan cleanup** — wired into FastAPI lifespan, UUID path parameters.

#### Blob Storage Manager

- **Phase 1** — data model, service foundation, migration.
- **Phase 2** — REST API routes and app wiring.
- **Phases 3–6** — frontend integration, composer tools, execution integration, schema inference.
- **Upload dedup, quota enforcement, and file cleanup.**

#### Secret Reference System

- **`SecretResolution` audit extension** — accepts `"env"` and `"user"` sources for web-originated secrets.
- **`resolve_secret_refs()` tree-walk** — recursive config replacement of `$secret{name}` references.
- **`ServerSecretStore`** and `WebSecretService` — chained resolution with allowlist enforcement, env-var boundary, fingerprint audit.
- **REST API, composer tools, execution integration, frontend wiring.**
- **Security hardening** — audit trail, fingerprints, leakage prevention, input validation.

#### Pipeline Execution Layer

- **Background pipeline runs** — `ExecutionServiceImpl` with WebSocket progress streaming and dry-run validation.
- **Cancel-vs-execute race closure** — atomic state transition preventing concurrent execution attempts.
- **Late WebSocket client seeding** — clients connecting after run start receive current state.

#### Pipeline Composer (LLM Tool-Use)

- **Frozen data models** — `SourceSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, `PipelineMetadata` with deep immutability.
- **Composition tools and YAML generator** — Sub-4B + 4C tool implementations.
- **`ComposerService` protocol** — LLM tool-use loop with prompts and message management (Sub-4D).
- **Wired to session routes** — composer integrated into session API.
- **Sub-4x hardening** — dual-counter loop guard, discovery cache, partial state recovery, rate limiting, tool registry.
- **Enhanced Stage 1 validation** — warnings, suggestions, and status tint.

#### Pipeline Inspector

- **Inspector UX overhaul** — EdgeSpec/NodeSpec fixes, graph readability improvements, version selector, catalog drawer.

#### Pipeline Composer MCP Server

- **`elspeth-composer` MCP server** — full pipeline composition toolset via Model Context Protocol. Tools for plugin discovery, pipeline state mutation, validation, YAML generation, and session persistence.
- **Pipeline-composer skill pack** — Claude Code skill for interactive MCP-driven pipeline building.
- **Pydantic model serialization** — fixed discovery tool responses.
- **Wave 4 tools** — `clear_source`, `explain_validation_error`, `list_models`, `preview_pipeline`.
- **Connection field sync** — when edges target outputs.
- **Path allowlist** — on `patch_source_options`, null argument guards.

#### Sink Failsink Pattern

- **`RowDiversion` and `SinkWriteResult`** — new contracts for per-row write failure routing.
- **`DIVERTED` outcome** — new terminal row state and `rows_diverted` counter.
- **`on_write_failure` mandatory config field** — `SinkSettings` requires explicit failure handling (`route_to`, `discard`, `fail`).
- **`BaseSink._divert_row()`** — with `FrameworkBugError` guard and protocol update.
- **`__failsink__` DIVERT edges** — DAG builder creates automatic diversion edges for sink failsink routing.
- **`validate_sink_failsink_destinations()`** — construction-time validation of failsink routing.
- **`SinkExecutor.write()` routing** — failsink dispatch on per-row write failure.
- **Hypothesis property tests** — partition-completeness and exactly-once routing invariants.

#### DAG Schema Propagation

- **`output_schema_config` as single source of truth** — populated for all node types (source, transform, gate, aggregation, coalesce) at construction time. `_assign_schema` refactored to only set `output_schema_config`, dropping the parallel dict write.

#### Frontend UX Refresh (A1-A7)

- **A1: Categorized file folders in blob manager** — files organized by category.
- **A2: Markdown and Mermaid rendering in chat** — rich content display with DOMPurify sanitization.
- **A3: Route validation errors visibly through chat** — errors surface in the conversation flow.
- **A4: Default 50/50 panel split** — balanced layout for graph and chat.
- **A5: Secrets button in chat toolbar** — moved with key icon for discoverability.
- **A6: Per-node validation indicators on graph** — visual status on each node.
- **A7: Three-state pipeline status indicator** — clear pipeline readiness feedback.
- **Validation indicator design tokens** — consistent visual language for validation states.
- **Validation orchestration extraction** — refactored to component layer.

#### Composer Agent Tooling (B1-B5)

- **Blob CRUD, structured validation, path redaction, pipeline diff** — agent-facing tools for the composer MCP server.

#### Composer Skill Pack Update

- **C1-C8, D1-D4 + deployment skill layer** — expanded skill definitions for composer interactions.

#### Web Group E

- **Unified file storage, blob refresh, inline source docs** — file handling consolidation.

#### Guard Symmetry Scanner

- **`enforce_guard_symmetry` CI tool** — detects write/read guard parity gaps (every Landscape write site must have a corresponding read guard). GitHub Actions workflow and allowlist support.

#### TokenRef Type

- **`TokenRef`** — bundled `token_id + run_id` frozen dataclass in `contracts/`. Replaces loose 2-tuple passing.
- **`AuditIntegrityError` loader guards** — Landscape read sites crash on corruption.
- **`coalesce_tokens` on TokenRef** — Landscape API accepts `TokenRef` directly.
- **`_validate_token_run_ownership` refactored** — accepts `TokenRef` instead of separate args.

#### Exception Hygiene

- **`TIER_1_ERRORS` constant** — canonical tuple of exception types for Tier 1 catch sites, applied across all layers.

#### Server Configuration

- **Default port 8451** — server config design with skill restoration.

### Fixed

#### P1 Bug Closure Campaign (~100+ bugs)

- **13 Landscape/Checkpoint/DAG integrity bugs** — audit write ordering, checkpoint restore invariants, DAG validation edge cases.
- **16 plugin transform bugs** — LLM response handling, multi-query field extraction, batch adapter identity, and miscellaneous isolates.
- **9 plugin source/sink bugs** — contract violations, atomicity gaps, and boundary validation.
- **10 engine orchestrator/processor/executor bugs** — execution loop invariants, processor state, executor edge cases.
- **7 web execution service bugs** — setup, race conditions, and state management.
- **3 checkpoint/coalesce integrity bugs** — resume state corruption and barrier restoration.
- **4 Landscape audit integrity bugs** — write guard gaps and recording consistency.
- **8 silent-failure and impossible-state validation bugs** — crash-on-invalid replacing silent skip.
- **3 LLM bugs** — empty choices audit gap, `tool_calls` fabrication, batch `finish_reason`.
- **7 web execution setup and contract silent-failure invariant bugs.**
- **9 sink phase ordering, expression parser coercion, and audit integrity bugs.**
- **4 `cluster:null-check` bugs** — retry `batch_id`, Chroma metadata, Azure audit, Annotated constraints.
- **3 `cluster:null-check` LLM bugs** — schema type erasure, content type validation.
- **8 `cluster:null-check` bugs** — NumPy float overflow, MCP contract drift, exporter field, LLM report condition.
- **6 `cluster:null-check` contract bugs** — NoneType inference, boolean guards, fabrication, userinfo leak, contract invariant.
- **7 pool shutdown, batch identity, and utils cluster bugs.**
- **4 SSRF gap, silent truncation, type crash, double-completion bugs.**
- **11 code review findings** — auth bypass, JSONL rollback, error narrowing.

#### Web Platform Hardening

- **Blob IDOR guard** — session deletion guard, orphan run cleanup.
- **21 code review findings** — across sessions, blobs, auth, execution.
- **17 code review findings** — FK constraints, 34 new tests.
- **6 code review findings** — Entra issuer, secret audit, cancel race, SNI, regex, fork timestamps.
- **3 code review findings** — `blob_ref` validation, fork guard, budget classification.
- **5 code review findings** — stranded runs, litellm dep, Chroma audit, WS race, shutdown iteration.
- **16 review findings** — across web epic subsystems.
- **Startup and auth regressions** — from code review integration.
- **Aggregation wiring, OIDC flow, and blob quota atomicity.**
- **Runtime routing fields** — for W1 output reachability check.

#### Plugin Hardening

- **Dataverse, RAG, and retrieval plugins** — 11 fixes from 5-agent review.

#### Deep Immutability

- **6 frozen dataclasses** — enforce deep immutability on mutable containers (contracts layer).
- **5 frozen dataclasses** — additional deep immutability enforcement.

#### Engine and Infrastructure

- **Terminal immutability in `complete_run()`** — Landscape enforces immutability on completed runs.
- **Tier 1 corruption guards** — added to MCP diagnostics and report analyzers.
- **Resource leaks closed** — weight validation added, error contracts hardened.
- **Non-finite float rejection** — at serialization and configuration boundaries.
- **`validate_input` unconditional** — removed opt-in flag; executor validates all input.
- **Validation error enrichment** — deterministic `repr_hash`, 8 test repairs.
- **6 pre-existing test failures** — across export, grades, and examples.
- **8 sweep findings** — dead code, redundant types, stale abstractions.

#### Code Review Synthesis

- **6-agent PR review findings** — metadata validation, `RunResult` hardening, consistency.
- **Failsink review** — cross-field checks, docstrings, test coverage.
- **6 correctness issues from PR review** — audit accuracy, fail-fast ordering, per-row diversion.
- **15 bugfixes from systematic code review** — expression parser, sink executor, Chroma, probes, bootstrap.
- **`hasattr` ban enforcement** — env isolation, type-check stubs.

#### Systematic Bug Sweep (Post-RC5-Cut — ~130 additional bugs)

- **36 bugs across 9 clusters** — audit integrity, silent failure, security, race conditions, resource leaks, freeze gaps, error handling, performance, web execution.
- **32 bugs across 6 groups** — audit integrity, validation, module hygiene, serialization, headers, engine contracts.
- **7 confirmed bugs** — audit integrity, state ordering, silent coercion.
- **8 type-safety and contract bugs** — from RC4 bug sweep.
- **7 quick-win bugs** — path anchoring, type contracts, dead params, stale docstring.
- **7 bugs + test infrastructure hardening** — phases 1-3.
- **6 Tier 1 audit integrity bugs** — inverted telemetry exception pattern.
- **5 validation-too-late bugs** — push constraints to construction boundaries.
- **6 P2 bugs** — config truncation, cache race, DNS ordering, schema tables, checkpoint integrity.
- **4 DAG/engine bugs** — plus telemetry circular import fix.
- **4 frozen dataclass / type safety bugs** — class-level shared-state hazard.
- **6 plugin/source/sink bugs** — config timing, type guards, boundary checks.
- **4 CLI/config/defensive-access bugs** — immutability, error messages, offensive guards.
- **AIMD zero-config guard** — `WebScrapeError` contract, plan doc types.

#### Tier Model & Exception Hygiene

- **7 unjustified `.get()` calls eliminated** — on Tier 1/2 data; replaced with direct access, `Counter`, freeze guard.
- **4 frozen DCs** — replaced shallow `MappingProxyType` wraps with `freeze_fields`.
- **Broad exception handlers narrowed** — `display_name` fabrication eliminated.
- **Typed `SchemaConfig` propagation** — `MappingProxyType` `to_dict` serialization fix.
- **Azure batch error shape probe** — split from field access; 3 reviewed-OK patterns documented.

#### Landscape & Audit Integrity

- **`complete_node_state()` terminal guard** — prevents terminal status overwrite.
- **`complete_batch()` terminal guard** — prevents terminal state overwrite.
- **DIVERTED in outcome validation** — added to Tier 1 read guards.
- **Terminal outcomes derived from `RowOutcome` enum** — closes DIVERTED gap in recovery.

#### Security

- **DSN credential scrubbing** — `_sanitize_dsn()` strips credentials from query parameters.

#### Web & Frontend Hardening

- **26 UX design review issues** — accessibility, contrast, touch targets (two review rounds: 14 + 12).
- **Mermaid SVG sanitization** — DOMPurify added to frontend.
- **Frontend type alignment** — types matched to backend schemas, system message rendering fixed.
- **Composer path resolution** — relative paths resolved against `data_dir`, not CWD.

#### Infrastructure & Contracts

- **`np.ndarray` in canonical JSON** — `sanitize_for_canonical()` handles NumPy arrays.
- **HTTP method serialization** — `to_dict()` serializes json/params for all methods.
- **Chroma distance function validation** — crash on missing metadata (offensive guard).
- **TOCTOU race fix** — seed race, `PluginNotFoundError` for validation, `str()` coercion removed.
- **Plugin discovery** — optional extras handling, JSONL MIME mapping.
- **DataverseClientError** — request metadata added.
- **LLM multi-query validation** — `WebSettings` required composer fields.
- **AIMD bootstrap** — fix after recovery, remove redundant finish-reason logging.
- **Session table DateTime columns** — timezone support added.
- **Review follow-ups** — TOCTOU guard, redundant or-None, TypeError catch.
- **8 stale code comments** — corrected across codebase.
- **221 logging errors** — from stale stdout capture in test suite.
- **Tier model fingerprints** — allowlist updates and `_state_response` entry.

### Changed

- **README web startup docs** — explicit instructions for `.[webui]` extra, building the frontend, `ELSPETH_WEB__SECRET_KEY`, creating a local auth user, and running the MVP locally.
- **Plugin manager singleton** — extracted from `cli.py` to `manager.py`.
- **532 mypy/ruff errors resolved** — across the full test suite.
- **CI hygiene** — format, mypy, stale allowlists from Sub-2 merge.
- **`_raise_if_invalid` extraction** — eliminates 3x error formatting duplication in manager.
- **`_make_span` extraction** — eliminates 7x no-op guard duplication in spans.
- **Stale schema mode references** — `fields: dynamic` → `mode: observed` across docs.

### Removed

- **errorworks test suite** — tests belong in the standalone package.
- **`archive/` directory** — all content preserved in git history.

### Tests

#### Test Hygiene Sweep

Systematic removal of low-value tests and replacement with behavioural gap-filling tests across all subsystems. Net result: fewer tests, better coverage of actual behavior.

- **Contracts** — removed 236 low-value tests, added 40 gap-filling tests.
- **Config** — removed 24 Pydantic default/assignment/frozen guarantee tests.
- **TUI** — removed 8 trivial import/existence checks, 11 TypedDict construction/duplicate tests; added 6 ExplainScreen loading tests, 3 node selection tests.
- **Telemetry** — removed 28 redundant tests, added 5 gap-fill tests.
- **MCP** — removed 8 trivial enum identity and method-existence tests; added 18 `get_error_analysis`/`get_llm_usage_report` tests.
- **Plugins** — removed 18 constructor passthrough/isinstance/decorator tests, 3 duplicate `PluginRetryableError` tests; added Truncate transform and `safety_utils` boundary tests.
- **Engine** — removed `test_run_status.py`, `test_diverted_counters.py`; added 11 orchestrator execution loop integration tests, partial purge failure invariant test.
- **Clock** — trimmed 9 redundant tests covered by property tests.
- **Models** — removed 54 low-value mutation-gap defaults tests.
- **Landscape** — consolidated 70 `where_exactness` tests into 36, 12 noncanonical validation error tests into 5; removed 4 stdlib-testing NaN guard tests.
- **Enums** — removed `test_enums.py`, `test_hookspecs.py`.

#### New Coverage

- **Azure Blob** — source and sink unit tests (config, CSV, JSON, JSONL, schema, audit) plus property-based tests.
- **DAG validation** — 15 error path tests.
- **Lineage** — 3 missing validation tests.
- **Builder** — validation gap tests, removed 34 low-value tests.
- **Web/Composer** — comprehensive `CompositionState` mutation and Stage 1 validation tests.
- **Web/Auth** — `ServerSecretStore` allowlist enforcement, env-var boundary, fingerprint audit tests.
- **Web/Prompts** — message isolation, ordering, context injection tests.
- **Buffer rollback** — strengthened to verify two-write scenario.

#### Post-Cut Coverage

- **8 coverage gaps closed** — IDOR, timezone, chroma metadata, schemas, mocks.
- **Source schema test strengthened** — verify `guaranteed_fields` round-trip.
- **Frontend test gaps** — from review, localStorage mock type annotation.
- **`error_edge_label`, control-flow exceptions, `CompatibilityResult`, `CONTRACT_TYPE_MAP`** — missing tests added, plus landscape error paths.
- **Mock spec enforcement** — `spec=LLMProvider` on unspec'd mocks, `spec=LandscapeRecorder` on resume failure test, `make_context()` landscape mock spec'd.
- **ChromaSink MagicMock contexts** — replaced with factories.
- **Batch transform tests** — `time.sleep` replaced with condition-based waits.
- **Autouse fixture narrowing** — scoped to telemetry dirs only.
- **Property test column strategy** — Python keywords filtered out.
- **`_on_write_failure` fixture** — replaced session-scoped fixture with explicit injection; hoisted to root conftest; `BaseSink.__init__` patched for integration tests.
- **`test_version_validation` reclassified** — `test_bootstrap_preflight` split.
- **`test_skill_drift`** — updated for `ValidationEntry` type change.

### Design Documentation

- **Web UX LLM Composer MVP** — design spec, 6 sub-specs, 6 sub-plans, program overview.
- **Sink failsink pattern** — design spec and 2-part implementation plan.
- **Fork-from-message** — sub-plan 04.
- **Composer hardening (Sub-4x)** — spec and implementation plan.
- **System Landscape spec** — platform-level audit trail.
- **Web test hygiene plan.**
- **Server config design.**
- **Frontend UX (A1-A7) implementation plan.**
- **VerifiedTokenRef implementation plan and review.**
- **Validation warning glossary** — added to CLI pipeline-composer skill.
- **Single-schema source-of-truth plan** — DAG `output_schema_config` propagation design.

## [0.4.1] (RC-4.1 — RAG Ingestion Pipeline)

Complete RAG ingestion story: ChromaSink for vector store population, pipeline `depends_on` for run sequencing, commencement gates for pre-flight go/no-go checks, and readiness contracts on retrieval providers. First pipeline-level orchestration primitives. Designed as a generic multi-stage pipeline pattern — RAG is the first consumer, but any plugin needing pre-populated external state can use the same mechanisms.

### Added

#### ChromaSink Plugin

- **ChromaSink** — new sink plugin writing pipeline rows into ChromaDB collections. Three `on_duplicate` modes: `overwrite` (upsert), `skip` (pre-filter existing IDs), `error` (pre-check and reject). Canonical content hash computed before write for audit integrity.
- **`FieldMappingConfig`** — explicit field mapping from row fields to ChromaDB concepts (`document_field`, `id_field`, `metadata_fields`). No convention-based defaults — operator declares exactly what goes where.
- **`DuplicateDocumentError`** — structured exception with `collection` and `duplicate_ids` (stored as immutable tuple) for `on_duplicate: error` mode.
- **ChromaDB metadata type validation** — metadata field values are validated as `str`, `int`, `float`, `bool`, or `None` at write time, before sending to ChromaDB. Invalid types (e.g. `dict`, `datetime`) crash with a `TypeError` naming the exact field, type, row index, and document ID.

#### Pipeline `depends_on` Mechanism

- **`depends_on` top-level config key** — declare pipelines that must run before the main pipeline starts. Each dependency is a fully independent pipeline run with its own `run_id`, Landscape records, and checkpoint stream.
- **`bootstrap_and_run()`** — reusable headless pipeline entry point in `cli_helpers.py` (L3). Handles secret resolution, passphrase handling, and directory creation. Injected into the dependency resolver via `PipelineRunner` protocol.
- **Circular dependency detection** — DFS cycle detector on canonicalized paths (`Path.resolve()`), with 3-level depth limit for nested dependencies.
- **Sequential execution** — dependencies run in declared order. `KeyboardInterrupt` propagates as-is (not wrapped in `DependencyFailedError`).
- **`DependencyRunResult`** — frozen dataclass with `run_id`, `settings_hash`, `duration_ms`, `indexed_at` for audit correlation.
- **Resume behaviour** — `elspeth resume` does NOT re-run dependencies. Fresh run required if dependencies need re-running.

#### Commencement Gates

- **`commencement_gates` top-level config key** — go/no-go conditions evaluated after dependencies complete, before the main pipeline starts.
- **`ExpressionParser` `allowed_names` extension** — gate expressions use the existing AST-whitelist parser with configurable namespace names (`collections`, `dependency_runs`, `env`). No `eval()`.
- **Pre-flight context** — assembled from dependency results, collection probes, and environment variables. Deep-frozen before gate evaluation (TOCTOU-safe). `env` excluded from Landscape audit snapshots to prevent secret leakage.
- **`collection_probes` explicit config** — operators declare which collections to probe. Probes assembled from explicit config, not auto-scanned from plugin configs.
- **`CommencementGateResult`** — frozen dataclass with `context_snapshot` deep-frozen via `freeze_fields()`.

#### Readiness Contract

- **`check_readiness()` on `RetrievalProvider` protocol** — returns `CollectionReadinessResult` (L0). Single-attempt, no retry. Called during `on_start()` after provider construction.
- **`ChromaSearchProvider.check_readiness()`** — collection count check with narrowed exception handling (connectivity errors only, not broad `except Exception`).
- **`AzureSearchProvider.check_readiness()`** — raw `httpx` count endpoint probe (not `AuditedHTTPClient`, which requires row-scoped `state_id`/`token_id` unavailable during `on_start()`).
- **RAG transform readiness guard** — `on_start()` checks both `reachable` and `count`. Raises `RetrievalNotReadyError` with `collection` and `reason` fields, with distinct messages for "empty" vs "unreachable".

#### Shared Infrastructure

- **`CollectionReadinessResult`** — unified frozen dataclass in L0 (`contracts/probes.py`) for all collection readiness checks. Used by probes, providers, and transforms.
- **`CollectionProbe` protocol** — L0 protocol for collection readiness probes, injectable into L2 engine without layer violations.
- **`ChromaConnectionConfig`** — shared Pydantic model for ChromaDB connection fields. Composed by `ChromaSinkConfig`, `ChromaSearchProviderConfig`, and `CollectionProbeConfig`. Collection name validated (min 3 chars, regex pattern).
- **`RetrievalNotReadyError`** — structured exception with keyword-only `collection` and `reason` fields.
- **`DependencyFailedError`** — structured exception with `dependency_name`, `run_id`, `reason`.
- **`CommencementGateFailedError`** — structured exception with deep-frozen `context_snapshot`.

#### Landscape Audit Trail

- **`preflight_results` table** — new Landscape table recording dependency runs and gate evaluations per pipeline run. `result_type` discriminator with `CheckConstraint`. Canonical JSON serialization via `deep_thaw()` + `canonical_json()`.
- **Readiness check outcomes recorded** — transform readiness results persisted alongside dependency and gate results.
- **Deferred recording pattern** — pre-flight results computed in `bootstrap_and_run()`, carried through `orchestrator.run()` as `PreflightResult`, recorded after `begin_run()`. Same pattern as `secret_resolutions`.
- **Dependency run correlation** — query run metadata links to indexing run via `run_id`, `settings_hash`, `indexed_at`. Auditor can trace: question → retrieved chunks → source documents → indexing decision → corpus state.

#### End-to-End Example

- **`examples/chroma_rag_indexed/`** — complete example: indexing pipeline (CSV → ChromaSink) + query pipeline (`depends_on` + commencement gate + RAG retrieval). Replaces the standalone `seed_collection.py` script with an audited pipeline.
- **CLI preflight wiring** — `elspeth run` now executes `depends_on` and commencement gates when configured.

### Fixed

#### Exception Hygiene Completion

- **3 overly-broad `except Exception` catches narrowed** to specific exception types in azure_batch, completing the exception narrowing sweep.
- **`batch_batch_timeout` double-prefix bug** — corrected in azure_batch per-row failure reason field. The prefix was applied twice, producing malformed reason strings.
- **Test asserting the bug** — corrected test that was asserting the buggy double-prefix behaviour rather than the correct single-prefix.
- **`TransformErrorCategory` and `TransformActionCategory` Literal gaps closed** — added missing category values that were valid at runtime but not in the type definitions.
- **HMAC-equivalent key pair filtering** — fingerprint property test now correctly filters HMAC-equivalent key pairs to avoid false failures.

#### Tier 1 Audit Integrity Hardening

- **`require_int()` utility** — Tier 1 int-field validator rejecting `bool` (Python's `isinstance(True, int)` footgun) and enforcing `min_value` bounds. Applied to 19 int fields across 13 audit dataclasses, plus `node_state_context`, `token_usage`, `batch_checkpoint`, `BufferEntry`, and `ResumePoint`.
- **TypedDict export records** — 15 typed shapes replacing `dict[str, Any]` in `LandscapeExporter._iter_records()`. `record_type` narrowed to `Literal` per record for mypy discriminated union support.
- **`CoalescePolicy` and `MergeStrategy` StrEnums** — replace bare strings in `CoalesceMetadata` and all call sites. Serialization-safe via `StrEnum`.
- **`Mapping[str, object]` write-path narrowing** — Tier 1 write paths in recorder and repositories narrowed from `dict[str, Any]` to `Mapping[str, object]` for tighter type safety.
- **`allow_nan=False`** — added to 6 `json.dumps()` calls in audit-path code, preventing NaN/Infinity from silently entering the Landscape.

### Changed

- **`ExpressionParser`** — now supports configurable `allowed_names` parameter (default `["row"]` for existing callers). Gate expressions use `["collections", "dependency_runs", "env"]`.
- **`ChromaSearchProviderConfig`** — refactored to compose `ChromaConnectionConfig` for shared validation. `to_connection_config()` method added.
- **`ElspethSettings`** — gains optional `depends_on`, `commencement_gates`, `collection_probes` fields.

---

## [0.4.0] (RC-4.0 — Plugins, Contracts, and Correctness)

Major feature release: Dataverse and RAG retrieval plugins, output schema contract enforcement, audit provenance boundary, freeze/serialize coherence, errorworks migration, and a 64-bug systematic sweep. Completes the RC-3.4 hardening sprint and delivers the first external-system plugin integrations. Also includes the agentic code threat model discussion paper (v0.1–v0.4) with MkDocs wiki and LaTeX build pipeline.

### Added

#### Dataverse Source and Sink Plugins

- **`DataverseSource`** — Microsoft Dataverse integration via OData v4 REST API. Supports structured OData queries and FetchXML with schema contracts. Pagination, SSRF validation, and rate limiting via the new `DataverseClient`.
- **`DataverseSink`** — upsert-only writes via PATCH with alternate key, idempotent for retries. Pre-processes all rows before HTTP calls.
- **`DataverseClient`** — pure protocol client handling authentication, pagination, SSRF validation, and rate limiting for the OData v4 API.
- **Shared utility extraction** — fingerprinting (`fingerprinting.py`) and strict JSON parsing (`json_utils.py`) extracted from `AuditedHTTPClient` into shared modules. 288 new tests.

#### RAG Retrieval Transform

- **`RAGRetrievalTransform`** — full retrieval-augmented generation transform with lifecycle management, process flow, and telemetry. Declared output schema config for downstream contract enforcement.
- **`RetrievalProvider` protocol** — L0 protocol with `RetrievalChunk` result type. Two implementations: `ChromaSearchProvider` (ephemeral/persistent/client modes, distance normalization) and `AzureSearchProvider` (score normalization, Tier 3 validation).
- **Query construction** — three modes: `field` (direct row field), `template` (string interpolation), `regex` (pattern extraction from row data).
- **Context formatting** — `numbered`, `separated`, and `raw` modes for assembling retrieved chunks into LLM context.
- **Shared template infrastructure** — extracted from LLM plugin for reuse by RAG query construction.
- **`PluginRetryableError`** — new base exception class. `LLMClientError` and `WebScrapeError` re-parented under it. Processor retry dispatch updated. New retrieval error categories added to `TransformErrorCategory`.
- **Example pipelines** — `examples/chroma_rag/` (standalone RAG) and `examples/chroma_rag_qa/` (RAG + LLM Q&A).

#### Output Schema Contract Enforcement

- **`_output_schema_config`** — new class attribute on `BaseTransform` with `_build_output_schema_config` helper. All field-adding transforms now declare their guaranteed output fields.
- **`FrameworkBugError` guard** — DAG builder crashes if a transform declares output fields but is missing `_output_schema_config`, preventing silent schema drift.
- **Integration tests** — full enforcement test and edge validation tests for output schema contracts.

#### Audit Provenance Boundary Enforcement

- **LLM audit metadata migration** — both batch and multi-query transforms now store audit fields in `success_reason` instead of polluting row data. Per-query provenance dicts collected and merged into `success_reason["metadata"]`.
- **`Call` return from `get_ssrf_safe()`** — `AuditedHTTPClient` now surfaces the `Call` object for audit correlation.
- **`payload_store` removed from `PluginContext`** — plugins access blob storage through `recorder.store_payload()` only, enforcing the provenance boundary.

#### Freeze/Serialize Coherence

- **Frozen container support in `contracts/hashing.py`** — canonical JSON serialization now handles `MappingProxyType`, `tuple`, and `frozenset` natively, resolving impedance mismatch between `deep_freeze` and hashing at L0.
- **Property tests** — hash equivalence (frozen == unfrozen), cross-module parity tests, frozen round-trip contract tests.
- **5 live bugs patched** — `MappingProxyType` NaN bypass, enum export, SSRF metadata, shallow thaw.
- **Thaw-refreeze elimination** — `plugin_context.record_call` no longer round-trips through thaw/refreeze. `ArtifactDescriptor` uses `deep_freeze` instead of shallow `MappingProxyType` wrap.

#### CI Enforcement

- **`enforce_freeze_guards.py`** — AST-based CI scanner detecting forbidden freeze patterns in `__post_init__`: bare `MappingProxyType` wraps (FG1) and `isinstance` type guards to skip freezing (FG2). Per-file allowlists with `max_hits`.
- **`enforce_mutable_annotations.py`** — CI linter detecting `list[]`/`dict[]`/`set[]` annotations on frozen dataclass fields. Allowlist for justified exceptions. *(Not yet merged — tracked for future implementation.)*
- **`freeze_fields()` promoted** to `contracts/freeze.py` as the canonical freeze utility. All freeze guards standardised.

#### web_scrape SSRF Allowlist

- **`allowed_hosts` configuration** — three-tier IP validation: `ALWAYS_BLOCKED_RANGES` (link-local, broadcast, multicast) → user allowlist (CIDR) → standard blocked ranges. Accepts `"public_only"` (default), `"allow_private"`, or explicit CIDR list. Threaded through redirect chain. 62 new tests.
- **Dual URL audit output** — both hostname and resolved IP URLs surfaced for audit comparison.

### Changed

- **`PipelineConfig` annotations** — `list`/`dict` fields changed to `Sequence`/`Mapping` to match frozen runtime types.
- **Field normalization mandatory** — removed `normalize_fields` toggle from `CSVSource`, `AzureBlobSource`, and `DataverseSource`. Header normalization always applied at the source boundary. Dunder name regression test added.
- **errorworks migration** — ChaosLLM, ChaosWeb, and ChaosEngine moved to external `errorworks` PyPI package (≥0.1.1). In-tree `chaosengine/`, `chaosllm/`, `chaosweb/`, `chaosllm_mcp/` directories deleted. Stale CLI subcommands removed.
- **Fabricated audit records removed** — `CallType.HTTP` records from `batch_replicate` validation and fabricated `variables_hash` sentinel from batch audit metadata.
- **Logger hygiene** — redundant logs removed, LLM `success_reason` audit metadata enriched.

### Fixed

#### RC4-Bugsweep (64 bugs across 13 clusters)

- **Broad-except swallowing framework errors** (6 bugs) — narrowed to specific exception types.
- **Exception type hygiene** (5 bugs) — replaced generic exceptions with domain-specific error types.
- **Dead code in exception handling** (4 bugs) — removed unreachable exception paths.
- **Missing audit on exception paths** (3 bugs) — added audit recording to previously silent failure paths.
- **Dataverse subsystem** (4 bugs) — sink pre-processing, source boundary validation.
- **Dataverse source Tier 3 boundary** (4 bugs) — trust boundary validation at OData response boundary.
- **`__post_init__` type guards** (4 bugs) — construction-time validation on checkpoint/engine dataclasses.
- **Freeze/immutability** (6 bugs) — `deep_freeze` Mapping support, shallow wrap elimination.
- **Tier 3 trust boundary validation** (5 bugs) — external system response validation.
- **LLM parallel execution audit integrity** (3 bugs) — concurrent audit recording correctness.
- **CI gate, retrieval, aggregation, verifier, CSV** (17 bugs) — cross-cutting correctness fixes.
- **IntegrityError propagation, `node_id` misattribution, buffer state corruption** (3 bugs).
- **Coalesce `rows_coalesced` double-increment** (1 bug) in timeout/flush path.
- **RAG subsystem** (3 bugs) — crash detection, resource cleanup, truncation budget.
- **ChromaDB distance type guard** (2 bugs) — crash on corrupt index instead of silent skip. Improved crash messages with collection name, doc ID, and remediation.

#### Additional Fixes

- **Plugin exception catch hygiene** (5 bugs) — pooling, search, sink, processor.
- **Exception type and chain hygiene** (5 bugs) — across contracts/engine/plugins.
- **Tier 1 checkpoint deserialization** — crash on corruption, don't coerce.
- **`AuditIntegrityError` misattribution** — prevented by outer exception handlers via dedicated error guard.
- **Unwrapped `record_call` SUCCESS paths** — all wrapped in `AuditIntegrityError`.
- **CLAUDE.md compliance** — defensive patterns, immutability, structlog, test types.

#### Mutation Testing

- **Checkpoint restore and WHERE clause exactness** — 71 new tests killing mutation survivors across 3 landscape repositories.
- **`canonical.py`** — 13 tests for None/NaT passthrough and numpy sanitization survivors.
- **`lineage.py`** — 3 tests for sink filter equality and terminal filtering survivors.

### Design Documentation

- **Dataverse design spec**: `docs/superpowers/specs/`
- **RAG retrieval design spec and implementation plan**: `docs/superpowers/specs/`, `docs/superpowers/plans/`
- **Output schema contract spec and plans**: `docs/superpowers/specs/`, `docs/superpowers/plans/`
- **Audit provenance boundary spec**: `docs/superpowers/specs/`
- **Freeze/serialize coherence spec**: `docs/superpowers/specs/`

---

## [0.3.4] (RC-3.4 — Systematic Hardening)

Systematic hardening sprint driven by 191-bug triage, mutation testing, and code quality sweep. Focus: audit integrity, deep immutability, construction-time validation, exception hygiene, and elimination of defensive anti-patterns. No new features — pure correctness and reliability work.

### Fixed

#### Audit Integrity & Tier 1 Hardening

- **PayloadNotFoundError domain exception** — `PayloadStore` protocol, `FilesystemPayloadStore`, and `MockPayloadStore` now raise `PayloadNotFoundError` instead of generic `KeyError`, preventing accidental catch by `except KeyError:` dict-lookup handlers. All five caller sites updated. PURGED paths now emit debug logs with `content_hash` for operational visibility.
- **PayloadIntegrityError → AuditIntegrityError** — `get_call_response_data` now catches `PayloadIntegrityError` and translates to `AuditIntegrityError` with run/call context, instead of letting raw integrity errors escape the landscape layer.
- **AuditIntegrityError for Tier 1 corruption** — Lineage queries, edge lookups, and purge grade updates now raise `AuditIntegrityError` instead of generic `ValueError` when encountering corrupt audit data.
- **Silent default=str fallback removed** — Journal serialization no longer silently coerces unserializable types via `default=str`. Non-serializable data now crashes immediately, exposing the upstream bug.
- **BatchCheckpointState tuple restoration** — `from_dict()` now restores tuple types after JSON round-trip instead of leaving them as lists, preserving Tier 1 checkpoint invariants.
- **Null-content LLM responses recorded** — Null-content responses are now recorded in the audit trail before raising, closing an audit gap where failed LLM calls left no trace.
- **Exception type hygiene** — `ValueError` replaced with `AuditIntegrityError` or `OrchestrationInvariantError` at 12 sites where the generic type misrepresented the failure category.
- **Tier 1 invariants in graph.py** — DAG graph now crashes on invalid source count, missing route labels, and defensive `.get()` patterns that masked corruption.
- **Programming-error guards in exporters** — Exporters and journal now raise `AuditIntegrityError` or `FrameworkBugError` instead of silently continuing on corrupt state.
- **Dead `ExecutionError.from_dict()` deleted** — Removed dead deserialization method; `TokenUsage.from_dict()` now rejects bool values that would silently coerce to int.

#### Deep Immutability & Frozen Dataclass Hardening

- **Central freeze/thaw utilities** — New `deep_freeze()` and `deep_thaw()` functions standardize immutability across all frozen dataclasses, replacing ad-hoc `deepcopy` calls.
- **deep_freeze recursion** — Now recurses into tuples, frozensets, and `MappingProxyType` contents, closing gaps where nested mutable containers survived freezing.
- **Mutable dict fields frozen** — All frozen checkpoint dataclasses now freeze mutable dict fields at construction, preventing post-construction mutation of Tier 1 data.
- **Category A mutable-frozen bugs** — Enforced deep immutability on 5 frozen dataclasses where mutable fields were exposed.
- **Category B mutable-frozen bugs** — Froze 5 additional DTOs with mutable internal state.
- **`slots=True` on all frozen dataclasses** — Added `slots=True` to `ResumeCheck`, `ResumePoint`, `RowDataResult`, `_GateEntry`, and all remaining frozen dataclasses that lacked it.
- **Contracts layer hardened** — Frozen sets, `deep_freeze` over `deepcopy`, `AuditIntegrityError` for checkpoint corruption.
- **HTTP DTO headers copied before freezing** — Prevents shared mutable header dicts from being modified after DTO construction.
- **Frozen constants and LineageResult** — Immutability hardened across engine constants and lineage query results.
- **Frozen/shared data structure enforcement** — Cleared `cluster:mutable-frozen` bug cluster.

#### Construction-Time Validation (`__post_init__`)

- **12 frozen dataclass types validated** — Added `__post_init__` validation enforcing invariants at construction time across contracts and engine types.
- **Remaining `cluster:missing-post-init` types** — Completed validation coverage for all frozen dataclasses that lacked construction-time checks.
- **NaN bypass and generator truthiness** — Fixed `__post_init__` validators that failed to detect NaN values and generators that evaluated truthy regardless of content.
- **Coalesce checkpoint DTO validation** — `CoalesceTokenCheckpoint` and `CoalescePendingCheckpoint` now enforce non-empty identifiers, non-negative timing, dict types, and disjoint branch keys.
- **Config-time validation** — Added validation for free-string fields, encoding, delimiters, and cross-field invariants at settings load time, clearing `cluster:config-validation`.

#### Exception Handling Hygiene

- **Exception chains preserved** — Replaced `from None` with `from exc` across 16 files, preserving diagnostic context in exception chains.
- **5 broken exception chains repaired** — Fixed `raise X from None` patterns in engine, plugins, and CLI that destroyed root-cause information.
- **22 broad `except Exception` catches narrowed** — Replaced overly broad catches with specific exception types, clearing `cluster:broad-except`.
- **Missing programming-error re-raises** — Added `FrameworkBugError`/`AuditIntegrityError` re-raise guards to telemetry `except` blocks that swallowed system errors.
- **Generic exceptions replaced** — Domain-specific error types (`AuditIntegrityError`, `GraphValidationError`, `OrchestrationInvariantError`) replace generic `ValueError`/`RuntimeError`, clearing `cluster:wrong-exception-type`.
- **Silent skips → explicit crashes** — 6 audit-gap bugs where code silently returned on invalid state now crash with invariant violation messages.

#### Defensive Pattern Removal

- **`hasattr()` banned unconditionally** — All 3 occurrences replaced with type-safe alternatives. `hasattr` is no longer allowlistable in the tier model enforcer.
- **Defensive `.get()` → required-fields validation** — `AggregationNodeCheckpoint.from_dict()` and other Tier 1 deserializers now validate required fields explicitly instead of using `.get()` with defaults.
- **Defensive access patterns removed** — Two passes across typed and Tier 1 data, replacing `.get(key, default)` with direct access on data we own.
- **CUSTOM header mode fail-closed** — Sink header mode `CUSTOM` now raises on unmapped fields instead of silently falling back to normalized names.

#### Data Fabrication Elimination

- **10 `cluster:fabrication` bugs fixed** — Replaced fabricated defaults (`None` → `0`, missing → empty string) with explicit validation or propagation of absence.
- **LLM batch, DB sink, and coalesce fabrication** — Eliminated silent default injection in three additional paths where missing data was replaced with invented values.

#### Audit-Gap Bug Fixes

- **4 TOCTOU races, silent blank rows, checkpoint invariant** — Fixed race conditions in concurrent audit writes and checkpoint state transitions.
- **3 credential leak, buffer corruption, record ordering** — Closed credential exposure in error messages, buffer mutation after read, and out-of-order audit record insertion.
- **2 rowcount validation, dict mutation** — Added rowcount assertions for fork/coalesce/expand writes; fixed in-place dict mutation in sequential multi-query.
- **4 row_data serialization, replayer Tier 1 reads** — Protected row_data from mutation during serialization; hardened replayer reads to crash on corruption.
- **Audit-gap silent failures recorded** — Error file download, malformed JSONL, and batch quarantine now record failures instead of silently dropping them.

#### Plugin & Engine Fixes

- **`_prepare_call_payloads` extracted** — Deduplicates payload preparation between `record_call` and `record_operation_call`.
- **`_make_checkpoint_after_sink_factory` extracted** — Deduplicates checkpoint-after-sink closure across orchestrator paths.
- **`dataclass_to_dict` tuple handling** — Now handles tuples correctly; fixed `has_retries` off-by-one comparison.
- **Azure client close-before-null** — `CallDataResult` discriminated type replaces ambiguous `None` return from Azure client operations.
- **Missing fingerprint key crashes** — Auth header fingerprinting now crashes on missing key instead of silently skipping; non-finite row indices recorded in batch stats.
- **Broken output port and shutdown timeout** — Batch mixin now crashes on broken output ports and respects shutdown timeout instead of silently continuing.
- **Purge grade updates wrapped individually** — Prevents stale grades after partial payload deletion.
- **Absent finish_reason accepted** — LLM responses with no `finish_reason` field (distinct from non-STOP values) are now accepted; Azure batch file ID guards added.
- **Per-query templates pre-compiled at init** — Separates structural errors (config, caught at startup) from operational errors (per-row, caught at render).
- **Coalesce `_completed_keys` for late arrivals** — Late-arriving tokens after resume are now correctly detected via completed-keys tracking.
- **Route-label enforcement at construction** — Moved from runtime lookup to DAG construction time; `get_route_label()` simplified.
- **`select_branch` KeyError wrapped** — Now raises `GraphValidationError` with context instead of bare `KeyError`.
- **`_get_node()` extracted in AggregationExecutor** — Unifies node validation, removes dead code.

#### Type Design

- **22 type-design bugs fixed** — 14 across engine, contracts, and plugins; 8 across engine, landscape, plugins, and verifier. Tightened field types, added missing validation, removed dead fields.
- **Checkpoint and call_data contracts hardened** — Type design improvements in checkpoint and call_data contract types.

#### Logging & Telemetry

- **stdlib logging → structlog** — Replaced in batch mixin, multi_query, azure_blob_source, and azure_batch.
- **Telemetry emitted before null-content raise** — Telemetry events are now emitted before raising on null-content LLM responses, closing an observability gap.
- **Journal payload load errors caught** — Journal now catches and translates `PayloadNotFoundError` and `OSError` from payload store with diagnostic context.
- **LLM finish-reason fail-closed** — Restructured `_finish_reason_error` from blocklist to allowlist (accept only `STOP` and absent). Unknown finish reasons now rejected as non-retryable errors.
- **Shutdown checkpoint skip logging** — `_checkpoint_interrupted_progress` emits structured `shutdown_checkpoint_skipped` warning with diagnostic context instead of silently returning.
- **Schema epoch directional guard** — `_sync_sqlite_schema_epoch` raises `SchemaCompatibilityError` on future epochs instead of silently downgrading.
- **Checkpoint recovery type annotation** — `_get_buffered_checkpoint_token_ids` parameter typed as `Checkpoint` instead of `Any`.
- **SQLite read-only audit inspection** — `LandscapeDB.from_url(..., create_tables=False)` no longer stamps `PRAGMA user_version`, preserving forensic access.

#### Test Infrastructure (ChaosLLM/ChaosWeb)

- **Malformed header overrides** — ChaosLLM and ChaosWeb now handle malformed header overrides gracefully instead of crashing the test server.
- **ChaosLLM template pre-compilation** — Templates pre-compiled at init for faster test execution.
- **Flaky purge test fixed** — `test_grade_update_failures_logged` used `capsys` which is unreliable when prior tests reconfigure structlog; switched to `structlog.testing.capture_logs()`.

### Changed

- `isinstance` allowlist compacted from flat entries into per-file rules with `max_hits` caps
- Code review findings remediated — frozen field access, `deep_thaw` frozenset, DTO validation

### Added

- **Agentic code threat model discussion paper** — Comprehensive research paper covering forward analysis for agentic security, control strength hierarchy, incentive misalignment analysis, ISM control citations, and ACF framework. Multiple revisions through v0.3 with LaTeX build pipeline and DTA brand guidelines.
- **`PayloadNotFoundError`** — Domain exception in `PayloadStore` protocol, replacing generic `KeyError` for missing payload lookups.
- **`CallDataResult` discriminated type** — Replaces ambiguous `None` returns from Azure client data operations.
- DAG validation tests for route-label and sink-map invariants
- Coalesce checkpoint unit tests for `_get_buffered_checkpoint_token_ids` and `restore_from_checkpoint` rejection paths

### Removed

- Dead lifecycle hooks, stale comments, unused imports across engine and plugins
- Dead code and process-tracking comments in ChaosLLM and ChaosWeb
- Dead code, tombstone comments, and process-tracking prefixes across codebase

### Tests

- 6 P0 mutation survivors killed across canonical, lineage, tokens, triggers, and coalesce
- 15 P1 mutation survivors killed across topology, lineage, tokens, triggers, coalesce, payload, exporter, executors, and outcomes
- 9 test-gap bugs closed with 25 new tests across executors, coalesce, DAG, sinks, and plugins
- 2 remaining test-gap bugs closed — purge command and MCP analyzer queries

---

## [0.3.3] (RC-3.3 — Architectural Remediation)

4-phase remediation sprint driven by full architecture analysis. Focus: audit integrity hardening, layer enforcement, and elimination of defensive-pattern violations.

### T10: LLM Plugin Consolidation

Collapsed 6 LLM transform classes (~4,950 lines) into a unified `LLMTransform` with provider dispatch, eliminating ~3,300 lines of duplication. Strategy pattern: `LLMProvider` protocol handles transport (Azure SDK vs OpenRouter HTTP), two processing strategies (`SingleQueryStrategy` / `MultiQueryStrategy`) handle row logic, shared `LangfuseTracer` handles tracing.

- Extracted shared infrastructure: `LangfuseTracer` (~600 lines deduplicated), `PromptTemplate` system, validation utilities
- Created `LLMProvider` protocol with `AzureLLMProvider` and `OpenRouterLLMProvider` implementations
- Single plugin registration: `plugin: llm` + `provider: azure|openrouter`; old names raise `ValueError` with migration guidance
- Deleted 5 old source files, updated 16 example YAMLs and 10 documentation files

### T17: PluginContext Protocol Split

Decomposed the god-object `PluginContext` (20+ fields) into 4 phase-based protocols in `contracts/contexts.py` — `SourceContext`, `TransformContext`, `SinkContext`, `LifecycleContext` — narrowing each plugin method signature to only the fields that pipeline phase actually needs. Concrete `PluginContext` structurally satisfies all 4 protocols; engine executors mutate concrete fields between steps while plugins see read-only views via protocol typing. 23 plugin files updated.

### T18: Orchestrator/Processor Decomposition

Pure extract-method refactoring of the two largest engine files, reducing maximum method size to ≤150 lines with no behavior change. Extracted 7 methods from `orchestrator/core.py` and 3 from `processor.py`. Introduced typed parameter bundles (`GraphArtifacts`, `RunContext`, `LoopContext`) and discriminated union types for transform/gate outcomes.

### T19: Landscape Repository Pattern

Refactored `LandscapeRecorder` from 8 mixins into 4 composed domain repositories — `RunLifecycleRepository`, `ExecutionRepository`, `DataFlowRepository`, `QueryRepository` — split by pipeline-phase domain. `LandscapeRecorder` is now a pure delegation facade (~91 public methods, zero logic).

### Plugins Restructure (SDA Alignment)

Reorganized the flat `plugins/` directory into 4 SDA-aligned subfolders: `infrastructure/` (shared base classes, clients, batching, pooling), `sources/`, `transforms/`, `sinks/`. 247 files changed, ~200 imports rewritten.

### Protocol Relocation (L3→L0)

Moved `SourceProtocol`, `TransformProtocol`, `SinkProtocol`, `BatchTransformProtocol`, and `GateResult` from `plugins/infrastructure/` (L3) to `contracts/` (L0). Eliminates the engine→plugins layer violation that forced L2 code to import from L3.

### Fixed

- **Pending coalesce resume gaps** — Added typed coalesce checkpoint DTOs, persisted pending coalesce state in checkpoint records, restored coalesce barriers on resume, and taught recovery to exclude buffered coalesce tokens from replay. Graceful shutdown can now resume fork/join pipelines without losing pending joins or replaying already-buffered rows.
- **Interrupted resume checkpoint ordering** — Resumed runs now rebase checkpoint sequence numbers from the previous resume point before writing fresh checkpoints, so a second interrupted resume continues from the newest durable progress marker instead of falling back to an older checkpoint.
- **SQLite schema compatibility posture** — Replaced the ad hoc `checkpoints.coalesce_state_json` required-column gate with an explicit SQLite schema epoch stamp via `PRAGMA user_version`, preserving intentional pre-1.0 schema breaks while keeping a clear future migration seam.
- **Buffered-only resume shutdown semantics** — Resume now honors a pre-set shutdown signal before any end-of-source aggregation/coalesce flushes, so buffered-only checkpoints are re-checkpointed for another resume instead of being flushed to sinks.
- **Frozen audit records** — Added `frozen=True, slots=True` to all 16 mutable audit record dataclasses in `contracts/audit.py`. Mutations now crash at the mutation site instead of silently corrupting the Tier 1 audit trail.
- **FrameworkBugError/AuditIntegrityError re-raise** — Added explicit re-raise before all broad `except Exception` handlers (13 sites across 7 files). System-level errors now always propagate. Structural AST test enforces bare `raise` pattern at all 17 guard sites.
- **Silent failure remediation** — Comprehensive review of error handling across LLM plugins and plugin infrastructure. Silent fallbacks converted to proper exceptions or `TransformResult.error()` with diagnostic context. Missing optional packages now raise `RuntimeError` with install instructions instead of silently degrading.
- **azure_batch silent passthrough** — `_process_single` else branch now raises `RuntimeError` instead of silently passing through unprocessed rows as "processed", matching the hardened pattern in `openrouter_batch`.
- **Assert removal** — Replaced 18 `assert` statements across 10 plugin files with explicit `if/raise RuntimeError`. Asserts are stripped by `python -O`, silently removing safety checks.
- **Truthiness checks** — Fixed 21 `if x:` / `x or default` patterns across 8 files that silently excluded valid zero values and empty strings. All replaced with explicit `is not None` checks.
- **LLM transform bugs** — Fixed limiter dispatch using wrong config attribute, `response_format` not passed to provider, `output_fields` not extracted from multi-query responses, NaN/Infinity not rejected in LLM JSON responses
- **Layer violations resolved** — Moved `ExpressionParser` from `engine/` to `core/`, `MaxRetriesExceeded` and `BufferEntry` to `contracts/`, created `RuntimeServiceRateLimit` in `contracts/config/`. 10 upward import violations → 0.
- **OpenRouter parallel query client race** — Parallel multi-query runs shared a cached `AuditedHTTPClient` by `state_id`; first query to finish destroyed the transport for siblings. Added reference counting so client closes only when last query releases it.
- **Aggregation BUFFERED lifecycle gap** — Triggering token on count-threshold flush skipped `BUFFERED` and went directly to terminal. Moved `BUFFERED` recording before `should_flush()` check so every aggregation token follows `BUFFERED` → terminal.
- **BatchReplicate quarantine audit gap** — Buffer-time recording changed from `CONSUMED_IN_BATCH` (terminal) to `BUFFERED` (non-terminal) for transform-mode aggregation, enabling per-token `QUARANTINED` recording when batch transforms quarantine individual rows.
- **KeywordFilter fail-closed on non-string values** — Security transform was silently passing non-string values in explicitly configured fields (fail-open). Now returns error with `reason='non_string_field'`.
- **Multi-query regressions from T10** — Restored field type validation against declared `output_fields` type/enum constraints; restored pooled execution with AIMD capacity backoff; fixed Pydantic schema missing `output_fields`; fixed `_output_schema_config` using unprefixed single-query fields.
- **LLM empty/whitespace content detection** — Azure and OpenRouter providers now raise `ContentPolicyError` for empty or whitespace-only content before `LLMQueryResult` construction.
- **LLM content-filter finish reason fail-open** — Unified single-query and multi-query `LLMTransform` paths now treat `finish_reason=content_filter` as `reason='content_filtered'` instead of recording provider-filtered fallback text as successful output.
- **Telemetry/Landscape hash divergence** — Telemetry hashes now read from recorded `Call` object instead of recomputing independently, eliminating divergence for datetime/Decimal/bytes/numpy payloads.
- **URL password fingerprint encoding** — Fingerprinting now decodes percent-encoding before HMAC, so fingerprint represents the actual secret, not the URL-encoded form.
- **TUI coalesce error crash on older records** — `_validate_coalesce_error` crashed with `KeyError` on pre-RC3.3 records. Added schema shape detection; older records render with degraded-format note.
- **Graceful shutdown end-of-source synthesis** — Interrupted runs no longer force `END_OF_SOURCE` aggregation flushes or resolve pending coalesces just because shutdown arrived after the current row.
- **Graceful shutdown resumability for buffered pipelines** — Interrupted aggregation/coalesce runs now persist a shutdown checkpoint before raising, so buffered state remains resumable even when no sink token was written yet.
- **CLI explain passphrase silently swallowed** (T4) — YAML parse errors when `--settings` was explicitly provided now exit with code 1 and clear error message.
- **MCP `diagnose()` quarantine count unscoped** (T5) — Was counting all historical runs; now scoped to last 24 hours, matching "what's broken right now?" purpose.
- **ChaosLLM MCP CLI broken** (T27) — Called nonexistent `serve()` instead of `run_server()`, masked by `# type: ignore` comments.
- **Azure AI tracing silent no-op** — Wired `_configure_azure_monitor()` into `LLMTransform.on_start()` with provider compatibility validation (Azure-only); replaced broad `except TypeError` with explicit `None` check so real SDK errors propagate.
- **Contract-level fixes** — `Token.run_id` false optional removed; `CoalesceFailureReason` TypedDict replaced with frozen dataclass (3 dead fields deleted, 4 fields made required); dead `version` parameter removed from `stable_hash()`; Call XOR invariant (`state_id` vs `operation_id`) now enforced at construction; `RawCallPayload.to_dict()` returns shallow copy per immutability contract; `SanitizedDatabaseUrl` rewrote DSN handling to use `urllib.parse` (keeping `contracts/` a leaf layer).
- **Code review remediation** — 4 critical (provider key validation split, `REPR_FALLBACK` row data state, `AuditIntegrityError` in `ExecutionError.from_dict()`, type-narrow `_convert_retryable_to_error_result`), 8 important, 6 suggestion fixes.
- **CI/CD failures resolved** — ruff lint/format, mypy stale `type: ignore`, contracts allowlist, tier model (31 stale fingerprints refreshed).

### Changed

- Extracted `contracts/hashing.py` — primitive-only `canonical_json`, `stable_hash`, and `repr_hash` (RFC 8785 + hashlib, no pandas/numpy). Breaks circular dependency between `contracts/` and `core/canonical.py`.
- Aggregation `on_error` is now required for aggregation transforms
- DTO mapper classes renamed from `*Repository` to `*Loader` to avoid confusion with new domain repositories
- **Test infrastructure overhaul (P0.5a–P4)** — 6-phase systematic hardening of the test suite, eliminating brittle coupling to internal constructors:
  - P0.5a–b: New factories (`make_recorder_with_run()`, `register_test_node()`, etc.) and refactored existing factories to delegate through them
  - P1: Replaced ~350 direct `PluginContext(...)` constructions across 53 files with centralized `make_context()` factory
  - P2: Replaced ~452 inline `LandscapeDB.in_memory()`/`LandscapeRecorder(...)` constructions across 76 files with factory calls. Net −715 lines
  - P3: Replaced ~529 lines of duplicated inline test plugin classes across 10 files with shared `tests.fixtures.plugins` imports
  - P4: Re-raise guards in telemetry/orchestrator/operation tracking, frozen evidence types (`ExceptionResult`, `FailureInfo`), aggregation DRY via `accumulate_row_outcomes()` + `ExecutionCounters`
- Resolved all 401 mypy errors across test suite — removed ~74 stale `# type: ignore` comments, added union-type narrowing guards, fixed module re-exports, wrapped `NewType` constructors, fixed protocol signatures (103 files)
- `PluginBundle` frozen dataclass replaces `dict[str, Any]` return from `instantiate_plugins_from_config()`, enabling mypy checking on all access sites
- Fingerprint primitives (`get_fingerprint_key()`, `secret_fingerprint()`) moved to `contracts/security.py` as stdlib-only implementations
- Redundant `.value` on `StrEnum` usage removed across checkpoint, landscape repositories, MCP, and tests
- Removed file-path header comments from 128 source files
- Azure safety transform consolidation (T14) — extracted shared batch infrastructure into `BaseAzureSafetyTransform` and `safety_utils.py`

### Added

- Typed coalesce checkpoint contracts (`CoalesceCheckpointState`, `CoalescePendingCheckpoint`, `CoalesceTokenCheckpoint`) plus CLI resume visibility for whether a checkpoint carries coalesce state
- **ADR-006**: Layer Dependency Remediation — documents the strict 4-layer model and CI enforcement strategy
- Full architecture analysis (23 documents covering all 13 subsystems)
- **Security posture brief** — Comprehensive document covering threat model, security controls, assurance evidence, and residual risk for ELSPETH v0.3.0
- **TYPE_CHECKING layer import detection** — `enforce_tier_model.py` CI gate now detects `TYPE_CHECKING` imports crossing layer boundaries as allowlistable findings
- **MCP server `_ToolDef` registry** replacing if/elif dispatch chain (T15)
- ~150 new tests across hardening, code review, and infrastructure phases

### Removed

- Dead code: `BaseLLMTransform` (3,473 lines, zero subclasses), `RequestRecord` dataclass, `TokenManager.payload_store` parameter, `populate_run()` (raw SQL bypass of `LandscapeRecorder`), LLM validation utilities (`render_template_safe`, `check_truncation`)
- ~21 low-value tests (vacuous assertions, mock-testing, implementation coupling)
- Superseded aggregation helpers replaced by shared `accumulate_row_outcomes()`

### Tests

- Full suite: approximately 10,500 tests — mypy/ruff/contracts all clean
- P0.5a–P4 test infrastructure overhaul: centralized factories, shared fixtures, eliminated ~1,700 lines of duplicated test boilerplate

---

## [0.3.0] - 2026-02-22 (RC-3.2)

### Highlights

- **Schema Contracts** — First-row-inferred field contracts propagated through the DAG and recorded in the audit trail
- **Declarative DAG Wiring** — Every edge explicitly named and validated at construction time
- **PipelineRow** — Typed row wrapper replacing raw dicts throughout the pipeline
- **Strict Typing at Audit Boundaries** — Every `dict[str, Any]` crossing into the Landscape audit trail replaced with frozen dataclasses, eliminating an entire class of silent data-corruption bugs
- **Test Suite v2** — Complete rewrite with 8,000+ tests across unit, property, integration, E2E, and performance layers
- **178-Bug Triage** — Systematic closure of 160+ bugs across 8 hardening phases

### Added

- Schema contract system: inference, propagation, sink header modes, and audit recording
- Typed DTOs at audit boundaries: `BatchCheckpointState`, `WebOutcomeClassification`, `NodeStateContext`, `CoalesceMetadata`, `AggregationCheckpointState`, `TokenUsage`, `GateEvaluationContext`, `AggregationFlushContext`, `CallPayload` protocol with typed request/response pairs
- `NodeStateGuard` context manager enforcing terminal-state invariants in all executors
- `detect_field_collisions()` utility preventing silent data overwrites across all transforms
- Azure Key Vault secrets backend with audit trail
- SQLCipher encryption-at-rest for the Landscape database
- WebScrape transform with SSRF prevention and content fingerprinting
- ChaosWeb fake server for stress-testing HTTP transforms
- Langfuse v3 tracing for LLM plugins
- Per-branch transforms between fork and coalesce nodes
- Graceful shutdown (SIGINT/SIGTERM) for run and resume paths
- DIVERT routing for quarantine/error sink paths

### Fixed

- **P0:** DNS rebinding TOCTOU in SSRF, JSON sink data loss on crash, content safety / prompt shield fail-open
- Frozen dataclass DTOs replacing `dict[str, Any]` at 10+ audit trail boundaries — eliminates runtime KeyError risk and tier-model allowlist entries
- `PluginContext.update_checkpoint()` replaced with `set_checkpoint()` (replacement semantics) — fixes P1 bug where dict merge lost checkpoint updates on restored batch state
- NaN/Infinity rejection at JSON parse and schema validation boundaries
- Resume row-drop, batch adapter crash, gate-to-gate routing crash
- Telemetry DROP-mode evicting newest instead of oldest events
- SharedBatchAdapter duplicate-emit race condition (first-result-wins preserved)
- AzureBlobSink multi-batch overwrite, CSVSource multiline skip_rows, JSONL multibyte decoding

### Changed

- Orchestrator, LandscapeRecorder, MCP server, and executors decomposed from monoliths into focused modules
- Checkpoint API typed: `get_checkpoint()` returns `BatchCheckpointState | None`, `set_checkpoint()` accepts typed state
- Pre-commit hooks scan full codebase (12 hooks, check-only)
- docs/ restructured from 792 files to 62 files
- All Alembic migrations deleted (pre-release, no users)

### Removed

- Gate plugin subsystem — routing is now config-driven only
- Beads (bd) issue tracker — migrated to Filigree
- V1 test suite (7,487 tests, 222K lines) — replaced by v2
- Dead plugin protocols (CoalesceProtocol, GateProtocol, PluginProtocol)

---

## [0.1.0] - 2026-02-02 (RC-2)

Initial release candidate. Core SDA pipeline engine with audit trail,
plugin system, and CLI.

## Historical Changelogs

- [RC-1 Changelog](CHANGELOG-RC1.md) — Initial framework build and hardening (Jan 12 – Feb 2, 2026)
- [RC-2 Changelog](CHANGELOG-RC2.md) — Sub-releases RC2 through RC2.5 (Feb 2 – Feb 12, 2026)

<!-- Comparison links — tags created at release time -->
[0.5.1]: https://github.com/tachyon-beep/elspeth/compare/v0.5.0-rc5.0...v0.5.1-rc5.1
[0.5.0]: https://github.com/tachyon-beep/elspeth/compare/v0.4.1-rc4.1...v0.5.0-rc5.0
[0.4.1]: https://github.com/tachyon-beep/elspeth/compare/v0.4.0-rc4.0...v0.4.1-rc4.1
[0.4.0]: https://github.com/tachyon-beep/elspeth/compare/v0.3.4-rc3.4...v0.4.0-rc4.0
[0.3.4]: https://github.com/tachyon-beep/elspeth/compare/v0.3.3-rc3.3...v0.3.4-rc3.4
[0.3.3]: https://github.com/tachyon-beep/elspeth/compare/v0.3.0-rc3.2...v0.3.3-rc3.3
[0.3.0]: https://github.com/tachyon-beep/elspeth/compare/v0.1.0-phase1...v0.3.0-rc3.2
[0.1.0]: https://github.com/tachyon-beep/elspeth/releases/tag/v0.1.0-phase1
