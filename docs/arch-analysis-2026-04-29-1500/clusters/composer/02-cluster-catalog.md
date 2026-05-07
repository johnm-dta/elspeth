# 02 — Cluster Catalog (composer cluster: web/ + composer_mcp/)

This catalog is one level deeper than 02-l1-subsystem-map.md §5 (web/) and §7 (composer_mcp/). Per Δ L2-3, sub-subsystem entries are capped at the immediate subdirectory tier; files >1,500 LOC are flagged as L3 deep-dive candidates and not summarised inline. Per Δ L2-7, each of the 7 SCC #4 members gets its own entry that explicitly cites cycle membership.

**Reading order:** composer_mcp/ (single transport entry, F1 headline) → 7 SCC members → 2 acyclic siblings (web/catalog, web/middleware) → web/frontend boundary record.

All `[ORACLE: ...]` citations resolve in `docs/arch-analysis-2026-04-29-1500/temp/l3-import-graph.json`. All intra-cluster edge citations of the form `(intra_cluster_edges[idx])` resolve in `temp/intra-cluster-edges.json`. All `[CITES KNOW-N]` markers resolve in `docs/arch-analysis-2026-04-29-1500/00b-existing-knowledge-map.md`.

**Notation legend (per validator Note 1):** `intra_cluster_edges[idx]` cites the entry whose `oracle_idx` field equals `idx` (i.e., the same index the L1 oracle uses), *not* the array position. The `intra-cluster-edges.json` array has 35 entries with `oracle_idx` values in {4, 5, 6, 39–47, 48–62, 63–76}. Every cited idx in this catalog resolves to exactly one entry; cross-reference by `jq '.intra_cluster_edges[] | select(.oracle_idx == N)'`.

---

## C1. composer_mcp/ — MCP transport over the composer state machine

**Path:** `src/elspeth/composer_mcp/`
**Responsibility:** Stateful MCP server that exposes the `web/composer/` state machine as an LLM tool surface (`mcp__elspeth-composer__*`); console script `elspeth-composer = "elspeth.composer_mcp:main"`.
**File count, LOC:** 3 files, 824 LOC (`server.py` 530, `session.py` 271, `__init__.py` 23).
**Tests:** `tests/unit/composer_mcp/{test_server.py, test_session.py}` — 2 files. **`tests/integration/composer_mcp/` ABSENT** (Δ L2-5 debt-flag candidate).

**Internal coupling:** Outside the cluster's web/* SCC; this is the second scope root and structurally separate from the cycle. composer_mcp imports across cluster boundary into `web/composer/` and `web/catalog/`:
- `composer_mcp → web/composer` weight 12, 3 sample sites including `composer_mcp/server.py:28,29` (intra_cluster_edges[6])
- `composer_mcp → web/catalog` weight 1, sample at `composer_mcp/server.py` (intra_cluster_edges[5])
- `composer_mcp → web` weight 1 (intra_cluster_edges[4]) — likely a config/dependency import.

**External coupling:** No outbound edges to other clusters at the package-collapse granularity. The transport layer reaches out only to its sister scope root (`web/`).

**Patterns observed:**
- `server.py:create_server(...)` and `main()` (server.py:530-line file) compose an MCP server whose tools are defined by `_build_tool_defs()` and dispatched by `_dispatch_tool()`/`_dispatch_session_tool()`. Tool argument JSON is sanitised by `_sanitize_error_for_client()` and `_ensure_serializable()` — the tier-3 boundary is explicit.
- `server.py:1–40` imports `CompositionState`, `PipelineMetadata` (from `web.composer.state`), tool entry points (from `web.composer.tools`), `generate_yaml`, `redact_source_storage_path`, and `CatalogService` (from `web.catalog.protocol`). **There is no parallel state machine in composer_mcp/**: it carries no `state.py` analogue, no fork of `tools.py`. composer_mcp is a thin MCP-protocol transport over the state machine that lives in `web/composer/`.
- `session.py` (271 LOC) defines `SessionManager` and `SessionNotFoundError` — these are MCP-session bookkeeping (tracking per-LLM-conversation state IDs), distinct from `web/sessions/` which persists user authentication sessions.

**Concerns:**
- `tests/integration/composer_mcp/` is absent. The MCP transport is unit-tested at the server/session level but no end-to-end test exercises an MCP tool round-trip; coverage gap surfaced as L2 debt candidate.
- The 530-LOC `server.py` carries 269 LOC of payload-typing scaffolding (the `_*Payload` TypedDicts) that mirrors web/composer schemas without sharing them; risk of schema drift between web HTTP responses and MCP tool responses. Surface only — no prescription per Δ L2-7-style discipline.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §7 (composer_mcp). Confirms F1 (`[ORACLE: edge composer_mcp → web/composer, weight 12]`) at the symbol level: composer_mcp imports `web.composer.{state, tools, yaml_generator, redaction}`. Closes L1 open-question Q2.

[CITES KNOW-G9] (the `ELSPETH_WEB__COMPOSER_EXPOSE_PROVIDER_ERRORS` opt-in is composer-specific and applies to both transports.)

**Confidence:** **High** — symbol-level evidence for transport claim; oracle-confirmed edge count; tests inspected.

---

## C2. web/ (top-level package files) — FastAPI app factory and shared infrastructure

**Path:** `src/elspeth/web/{__init__.py, app.py, async_workers.py, config.py, dependencies.py, paths.py, validation.py}`
**Responsibility:** FastAPI app composition (the `create_app(...)` factory in `app.py`), shared settings (`WebSettings`), async-worker dispatch helpers, path constants, and request-scope dependencies.
**File count, LOC:** 7 top-level Python files, **1,072 LOC** total. Largest: `app.py` 652, `config.py` 233, `validation.py` 56, `dependencies.py` 51, `paths.py` 50, `async_workers.py` 29, `__init__.py` 1.
**Tests:** `tests/unit/web/{test_app.py, test_config.py, test_dependencies.py, test_paths.py}` — 4 files at root.

**Internal coupling:** **Member of SCC #4 with web/{auth, blobs, composer, execution, secrets, sessions}; acyclic decomposition not possible.** This top-level package node is the cycle's *anchor* — `app.py` outwardly imports every sub-package's `routes.py` and service factory, while sub-packages reach back via `from elspeth.web.config import WebSettings`, `from elspeth.web.async_workers import run_sync_in_worker`, `from elspeth.web.paths import ...`. Direct outbound edges to sub-packages: `web → web/auth` weight 6 (intra_cluster_edges[40]), `web → web/secrets` weight 5 (idx 46), `web → web/sessions` weight 5 (idx 47), and similar to blobs/catalog/composer/execution/middleware. Inbound from sub-packages: `web/auth → web` weight 6 (idx 48), `web/execution → web` weight **15** (idx 63 — heaviest intra-cluster edge), `web/secrets → web` weight 6 (idx 68), and others.

**External coupling:** `web → plugins/infrastructure` weight 1 (oracle_idx 39, sample `web/dependencies.py:48`). This is the only outbound edge from the package root — request-scope dependency injection touches the plugin infrastructure registry.

**Patterns observed:**
- `app.py:create_app(...)` is the staging deployment entry per [CITES KNOW-G6]: `uvicorn elspeth.web.app:create_app --factory ...`. The factory wires every sub-package router into a single FastAPI instance.
- `config.py` defines `WebSettings` (Pydantic + Dynaconf) — imported by every sub-package; this is the structural reason for the inward leg of the cycle.
- `async_workers.py:run_sync_in_worker(...)` (29 LOC) is referenced by sub-packages that need to dispatch synchronous work off the event loop.
- `app.py:128 except (SQLAlchemyError, OSError) as cleanup_exc:` (in `_periodic_orphan_cleanup`) is one of 269 R6 silent-except findings flagged by `enforce_tier_model.py`; not a blocker but a debt candidate.

**Concerns:**
- The 1,072 LOC of "infrastructure" the SCC's "web" node represents is concentrated in `app.py` (652 LOC). At ~60% of the package-root LOC, app.py is the cycle's structural pinch point; if the SCC ever needs decomposition, app.py is where decomposition pressure will land.
- `validation.py` (56 LOC) imports from `web.validation` (per `config.py:from elspeth.web.validation import ...`) — confirm at L3 whether this is a re-export pattern or a single-module split.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §5 (web). Anchors §7.5 F4's SCC analysis (the "web" node in the 7-node SCC = these 7 files).

[CITES KNOW-G6] [CITES KNOW-G7] (staging entry point and SPA mount).

**Confidence:** **High** — file enumeration and SCC anchor confirmed by oracle; entry-point name confirmed in `pyproject.toml` referenced by KNOW-G6.

---

## C3. web/auth/ — Authentication providers and FastAPI auth dependency

**Path:** `src/elspeth/web/auth/`
**Responsibility:** Pluggable authentication providers (local SQLite + bcrypt + JWT, OIDC with JWKS discovery, Azure Entra ID) plus the FastAPI `get_current_user` dependency that extracts `UserIdentity` from Bearer tokens.
**File count, LOC:** 8 files, 1,224 LOC. `entra.py`, `local.py`, `middleware.py`, `models.py`, `oidc.py`, `protocol.py`, `routes.py`, `__init__.py`.
**Tests:** `tests/unit/web/auth/` — 7 unit-test files.

**Internal coupling:** **Member of SCC #4 with web/{web, blobs, composer, execution, secrets, sessions}; acyclic decomposition not possible.** Inbound from cycle: `web → web/auth` weight 6 (intra_cluster_edges[40]), `web/execution → web/auth` weight 6 (idx 64), `web/blobs → web/auth` weight 2 (idx 51). Outbound: `web/auth → web` weight 6 (idx 48), `web/auth → web/middleware` weight 1 (idx 49).

**External coupling:** No direct outbound edges to other clusters at package-collapse granularity.

**Patterns observed:**
- Three authentication providers behind one `AuthProvider` protocol (`protocol.py`): local (`local.py` — bcrypt, PyJWT), OIDC (`oidc.py` — JWKS discovery, JWT signature validation), Entra (`entra.py` — composes `JWKSTokenValidator` and adds tenant-specific claim checks). The composition pattern (entra → oidc → jwks-validator) is explicit at the docstring level.
- `middleware.py` is a *FastAPI dependency*, not ASGI middleware (its docstring is explicit on this point: "All protected routes use this dependency function").
- `models.py:UserIdentity` and `UserProfile` are frozen dataclasses with scalar-only fields — per CLAUDE.md they need no `__post_init__` freeze guard, and the docstring confirms this.
- `routes.py` exposes `/api/auth/{login,register,token,config,me}`; `POST /login` is conditionally registered ("only available when auth_provider is 'local'" per docstring).

**Concerns:**
- Active P1 bugs at `src/elspeth/web/auth/local.py` and `oidc.py` are tracked outside this analysis (filigree IDs `elspeth-fb0cc6d507`, `elspeth-4db1461d97`, `elspeth-93b6659aa3`). The catalog records existence; remediation is task-tracker scope, not archaeology scope.
- Per the `tests/unit/web/auth/` count (7 files), each provider has its own test file, but observation of integration test coverage (the routes layer, end-to-end) is deferred — only `tests/integration/web/test_execute_pipeline.py` exists, which does not target auth.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §5 (web/, sub-area `auth/`). No KNOW-A* claim covers auth/ specifically — the existing-knowledge map predates web-UI maturity.

**Confidence:** **High** — file enumeration via `ls`; provider model confirmed by docstrings; test directory confirmed by `find`.

---

## C4. web/blobs/ — Blob upload/download with quota and content-hash integrity

**Path:** `src/elspeth/web/blobs/`
**Responsibility:** Blob ingestion service (POST/finalize/download flows), quota enforcement, content-hash verification, MIME sniffing.
**File count, LOC:** 6 files, 1,952 LOC. `protocol.py`, `routes.py`, `schemas.py`, `service.py`, `sniff.py`, `__init__.py`.
**Tests:** `tests/unit/web/blobs/` — 5 unit-test files.

**Internal coupling:** **Member of SCC #4.** Outbound: `web/blobs → web` weight 1 (intra_cluster_edges[50]), `web/blobs → web/auth` weight 2 (idx 51), `web/blobs → web/sessions` weight 4 (idx 52). Inbound: `web → web/blobs` weight (in 41–45 range, see oracle), `web/composer → web/blobs` weight 5 (idx 56), `web/execution → web/blobs` weight 5 (idx 65), `web/sessions → web/blobs` (intra_cluster_edges).

**External coupling:** No direct outbound edges to other clusters. Blobs operates entirely within the web cluster.

**Patterns observed:**
- `service.py:BlobServiceImpl` is the central class. `service.py:content_hash(data)` and `_validate_finalize_hash(...)` lock in integrity (the audit-trail "hashes survive payload deletion" rule from CLAUDE.md applies at this surface).
- `_assert_blob_run_same_session(...)` and `_source_references_blob(...)` enforce session/run scoping — blobs are scoped to a session and validated against the session's owning run before access.
- `sniff.py` separates MIME detection from the service surface; `routes.py` handles HTTP-side parsing.

**Concerns:**
- The `web/blobs → web/sessions` weight-4 edge means session lifecycle is tightly coupled to blob lifecycle. Cascade-deletion semantics (when a session is closed, are its blobs orphaned?) — observed in `app.py:_periodic_orphan_cleanup` (the silent-except site at app.py:128). Confirmed cleanup path; see "session-db-reset" guide (`docs/guides/session-db-reset.md`, untracked at session start) for the operator-facing version.
- Hash integrity tests are likely in `tests/unit/web/blobs/`; specific test name(s) confirming the audit-trail "hashes survive payload deletion" invariant are not enumerated here — invariant asserted in code only at this depth, recorded as L2 debt-flag candidate (would benefit from an explicit named test).

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §5. No KNOW-A* claim for blobs.

**Confidence:** **High** for structural coupling claims; **Medium** for the test-invariant claim (not specifically enumerated).

---

## C5. web/composer/ — LLM-driven pipeline composer state machine and tool surface

**Path:** `src/elspeth/web/composer/`
**Responsibility:** Stateful pipeline composer — owns the `CompositionState` and `PipelineMetadata` types, the LLM tool surface that mutates them (`tools.py`), the YAML generator, the semantic validator, and the per-conversation progress sink.
**File count, LOC:** 11 top-level Python files (12 with `skills/` sub-package), **8,189 LOC top-level / 8,274 LOC including `skills/`**. Files: `_producer_resolver.py`, `progress.py`, `prompts.py`, `protocol.py`, `redaction.py`, `_semantic_validator.py`, `service.py`, `state.py`, `tools.py`, `yaml_generator.py`, `skills/`.
**Tests:** `tests/unit/web/composer/` — 13 unit-test files (the cluster's largest test group).

**>>> L3 deep-dive candidates (Δ L2-3 — flagged, NOT summarised inline):**
- `tools.py` — **3,804 LOC** — the single largest file in the entire ELSPETH tree. LLM tool definitions and dispatch, including the `_dispatch_tool` shape that composer_mcp re-exports.
- `state.py` — **1,710 LOC** — the `CompositionState` and `PipelineMetadata` types, all mutation primitives.
Both are listed in 02-l1-subsystem-map.md §11.3.

**Internal coupling:** **Member of SCC #4 (cluster-internal central type).** This sub-package is the most heavily-imported in the cluster:
- `composer_mcp → web/composer` weight 12 (the F1 edge, intra_cluster_edges[6])
- `web/sessions → web/composer` weight **15** (intra_cluster_edges[74]) — sessions persist composer drafts
- `web/execution → web/composer` weight 9 (intra_cluster_edges[66]) — execution validates composer-built pipelines
- `web/composer → web` weight 4 (idx 55)
- `web/composer → web/blobs` weight 5 (idx 56)
- `web/composer → web/catalog` weight 4 (idx 57)
- `web/composer → web/composer/skills` weight 2 (idx 58 — sub-package import, the markdown-skill loader)
- `web/composer → web/sessions` weight 5 (idx 59)
**The state machine in `web/composer/state.py` is consumed by composer_mcp (transport), sessions (persistence), AND execution (validation).** This is an even stronger architectural claim than F1: web/composer is a cluster-internal central type, not just an MCP-transport target.

**External coupling:** `web/composer → plugins/infrastructure` weight **22** (oracle_idx 54, sample sites `web/composer/_semantic_validator.py:47–53`) — the heaviest single outbound edge from the cluster. The semantic validator reads plugin contracts to validate composer output against runtime plugin signatures.

**Patterns observed:**
- `service.py:ComposerServiceImpl` is the HTTP-side composer service; `service.py:_collect_required_paths`, `_build_tool_required_paths_index`, `_find_missing_required_paths` are validation helpers that interrogate the composer state for required-field completeness.
- `protocol.py` defines `ComposerResult`, `ComposerService` (Protocol), and a typed exception hierarchy: `ComposerServiceError → {ComposerConvergenceError, ComposerPluginCrashError, ToolArgumentError}`.
- `_semantic_validator.py` is the bridge to `plugins/infrastructure` (weight-22 edge); it imports plugin contracts at file-top and validates composer state against them.
- `redaction.py:redact_source_storage_path(...)` is imported by composer_mcp (server.py:1) — consistent secret-handling between transports.

**Concerns:**
- 8,274 LOC across 12 files with 67% of LOC concentrated in two files (tools.py + state.py = 5,514 LOC) is the cluster's quality-risk concentration. Symptomatic of "tool registry + state machine in one sub-package" pattern; whether this is essential complexity or a candidate for splitting is an architecture-pack question (per Δ L2-7), not an archaeology question.
- The L1 catalog flagged tools.py and state.py as composite-at-L2-depth candidates per the Δ4 heuristic. With 11 top-level files and 8,274 LOC, web/composer/ itself qualifies as composite at L2 depth. **L3 candidate: composite at L2 depth.** Catalog stops here.
- 13 unit-test files is substantial coverage; explicit test paths verifying the F1 transport-equivalence (web/composer behaviour matches composer_mcp behaviour for shared tools) are not enumerated at this depth.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §5 (web/, sub-area `composer/`). Anchors §7.5 F1 (`[ORACLE: composer_mcp → web/composer, weight 12]`) and refines it: composer is consumed by sessions (weight 15) and execution (9) as well, beyond the F1 transport relationship.

[CITES KNOW-G9] (composer-specific provider-error opt-in switch).

**Confidence:** **High** for structural claims (file counts, edge weights, L3-deep-dive flags); **Medium** for "essential vs accidental" framing — that's an architecture-pack call.

---

## C6. web/execution/ — HTTP-side run-execution service

**Path:** `src/elspeth/web/execution/`
**Responsibility:** HTTP API for triggering pipeline runs, validating composer-built YAML, surfacing run progress and diagnostics, and producing discard summaries.
**File count, LOC:** 11 files, 3,748 LOC. Largest: `service.py` (the run-execution orchestration), then `validation.py`, `_semantic_helpers.py`, `routes.py`, `schemas.py`, `progress.py`, `errors.py`, `diagnostics.py`, `discard_summary.py`, `protocol.py`, `__init__.py`.
**Tests:** `tests/unit/web/execution/` — 8 unit-test files; `tests/integration/web/test_execute_pipeline.py` is the cluster's only integration test and targets this sub-package.

**Internal coupling:** **Member of SCC #4.** Heaviest reach-back in the cluster: `web/execution → web` weight **15** (intra_cluster_edges[63]) — the heaviest intra-cluster edge. Other heavy outbound: `web/execution → web/composer` weight 9 (idx 66), `web/execution → web/sessions` weight 7 (idx 67), `web/execution → web/auth` weight 6 (idx 64), `web/execution → web/blobs` weight 5 (idx 65).

**External coupling:** Three of the cluster's six outbound edges originate here:
- `web/execution → .` (root `elspeth` package) weight 3 (oracle_idx 60, sample sites `web/execution/service.py:30`, `:805`, `validation.py:24`) — *unusual at this granularity*. Likely `from elspeth import <something>` referencing a public symbol re-exported at the package root; flagged for synthesis pass.
- `web/execution → plugins/infrastructure` weight 4 (oracle_idx 61, samples `_semantic_helpers.py:66,67`, `validation.py:30`).
- `web/execution → telemetry` weight 1 **(conditional, oracle_idx 62, sample `service.py:781`)** — the cluster's only conditionally-imported cross-cluster dependency. Telemetry is feature-gated.

**Patterns observed:**
- `service.py:ExecutionServiceImpl` is the central class. `service.py:_sanitize_error_for_client(exc)` and `_resolve_yaml_paths(pipeline_yaml, data_dir)` are tier-3-boundary helpers — operator-supplied YAML and external errors are sanitised before crossing the HTTP boundary.
- `service.py:805` is past the file's first 805 lines — file is dense; not flagged as L3-deep-dive (under 1,500 LOC), but high density.
- The conditional `→ telemetry` import (`service.py:781`) implements optional telemetry: telemetry is observed-but-not-required at this site. Pattern-consistent with CLAUDE.md's "telemetry is operational visibility, not legal record" framing.

**Concerns:**
- The `web/execution → .` edge at oracle granularity is the most unusual in the cluster — every other outbound edge has a meaningful target (plugins/infrastructure, telemetry, sub-cluster). Without entering files, the catalog cannot say what symbol(s) at the root package are imported. Surface for synthesis.
- 3,748 LOC across 11 files is the cluster's third-densest sub-package after composer and sessions. The execution service's depth (`service.py:805`+ at evidence sites) suggests `service.py` is approaching the deep-dive threshold (under 1,500 by tier-model run, but the catalog has not measured `service.py` LOC directly).

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §5. No KNOW-A* claim for execution/.

**Confidence:** **High** for edge claims (oracle); **Medium** for `service.py` density observation (line numbers cited but file LOC not measured at this depth).

---

## C7. web/secrets/ — Server-stored secrets with per-user and server scopes

**Path:** `src/elspeth/web/secrets/`
**Responsibility:** Encrypted secret storage with two scopes (server-wide and per-user), accessed by composer/execution at LLM-provider configuration time.
**File count, LOC:** 6 files, 1,011 LOC. `routes.py`, `schemas.py`, `server_store.py`, `service.py`, `user_store.py`, `__init__.py`.
**Tests:** `tests/unit/web/secrets/` — 5 unit-test files.

**Internal coupling:** **Member of SCC #4.** `web → web/secrets` weight 5 (intra_cluster_edges[46]); `web/secrets → web` weight 6 (idx 68). Likely consumed by composer (provider credentials) and execution (run-time provider auth), but those edges are aggregated at the package-collapse granularity into the higher-weight composer/execution → web edges, not enumerated as direct secrets-consumer edges in the oracle.

**External coupling:** No direct outbound edges to other clusters.

**Patterns observed:**
- `server_store.py:ServerSecretStore` and `user_store.py` (per its file name) implement the two-scope split.
- `_is_reserved(name)` (server_store.py) protects reserved namespaces — namespace-based access control rather than capability-based.
- `routes.py` and `schemas.py` separate HTTP surface from internal types.

**Concerns:**
- None observed at L2 depth that aren't captured in the SCC analysis.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §5. No KNOW-A* claim.

**Confidence:** **Medium** — file enumeration confirmed; specific secret-rotation invariants would need L3 inspection.

---

## C8. web/sessions/ — Session and authentication-token persistence

**Path:** `src/elspeth/web/sessions/`
**Responsibility:** SQLAlchemy-backed session model (the user-session-and-bearer-token persistence layer; not to be confused with composer_mcp's per-conversation sessions).
**File count, LOC:** 9 files, 4,080 LOC. `converters.py`, `engine.py`, `models.py`, `protocol.py`, `routes.py`, `schema.py`, `schemas.py`, `service.py`, `__init__.py`.
**Tests:** `tests/unit/web/sessions/` — 9 unit-test files (1:1 file-to-test ratio).

**Internal coupling:** **Member of SCC #4.** Sessions is the most densely-connected sub-package in the cluster — 5 inbound and 6 outbound intra-cluster edges:

- *Inbound* (others → sessions): `web → web/sessions` weight 5 (intra_cluster_edges[47]); `web/blobs → web/sessions` weight 4 (idx 52); `web/composer → web/sessions` weight 5 (idx 59); `web/execution → web/sessions` weight 7 (idx 67); `web/secrets → web/sessions` weight 1 (idx 70).
- *Outbound* (sessions → others): `web/sessions → web/composer` weight **15** (idx 74) — the joint-heaviest intra-cluster edge, semantically distinct from the other weight-15 because it is data-flow-driven (sessions persists composer drafts), not cycle-driven; `web/sessions → web` weight 3 (idx 71) — the inward-leg cycle edge to the package root; `web/sessions → web/auth` weight 2 (idx 72); `web/sessions → web/blobs` weight 2 (idx 73); `web/sessions → web/execution` weight 2 (idx 75); `web/sessions → web/middleware` weight 2 (idx 76).

The bidirectional density (sessions → composer, then composer → sessions weight 5 back; similarly for blobs, execution) reflects sessions' role as the persistence layer that lifecycle-manages other sub-packages' state.

**External coupling:** No direct outbound edges to other clusters.

**Patterns observed:**
- `service.py:SessionServiceImpl` is the central class. `_assert_state_in_session(...)` enforces the invariant that mutated state must belong to the current session.
- The split between `schema.py` (likely DB schema) and `schemas.py` (likely Pydantic) is uncommon — two distinctly-named modules suggests deliberate separation of persistence shape from API shape.
- `engine.py` (SQLAlchemy engine) and `converters.py` (value converters) are persistence-layer details.
- `web/sessions → web/composer` weight 15 means **sessions persists composer drafts** (CompositionState instances). This is the central data-flow finding for the cluster: composer drafts have a persistence boundary, and that boundary is sessions.

**Concerns:**
- 4,080 LOC across 9 files (avg 453 LOC/file) is the cluster's second-densest sub-package after composer. No single file flagged as L3 deep-dive at the L1 pass, but `service.py` density warrants verification at L3.
- `tests/unit/web/sessions/` 9 files at 1:1 ratio is structurally good; verifying that invariant tests (e.g. session-state cascade-deletion) are present is L3 scope.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §5. No KNOW-A* claim.

**Confidence:** **High** for coupling claims; **Medium** for service density (no direct LOC measurement of `service.py` at L2 depth).

---

## C9. web/catalog/ — Plugin catalog service (acyclic — outside SCC)

**Path:** `src/elspeth/web/catalog/`
**Responsibility:** Read-only plugin catalog service — surfaces plugin metadata (sources, transforms, sinks) to the composer and HTTP clients.
**File count, LOC:** 5 files, 407 LOC. `protocol.py`, `routes.py`, `schemas.py`, `service.py`, `__init__.py`.
**Tests:** `tests/unit/web/catalog/` — 3 unit-test files.

**Internal coupling:** **NOT in SCC #4 — acyclic.** Inbound from cluster: `web/composer → web/catalog` weight 4 (intra_cluster_edges[57]), `composer_mcp → web/catalog` weight 1 (idx 5). No outbound edges into the cycle: catalog reads plugin metadata and *does not* import back into the package root or sub-packages, which is what keeps it out of the SCC.

**External coupling:** `web/catalog → plugins/infrastructure` weight 3 (oracle_idx 53, sample sites `web/catalog/service.py:8,9`).

**Patterns observed:**
- `service.py:CatalogServiceImpl` is the central class — exposes plugin metadata read from the registry.
- The acyclic structure is *deliberate*: catalog is consumed by composer (for tool-set introspection) and composer_mcp (for the same), but does not need to reach back into shared types because it owns its own schemas.
- No pre-L1 KNOW-* claim; per filigree there is an open P2 bug `elspeth-dcf12c061b` against `CatalogServiceImpl.get_schema(...)` that is task-tracker scope, not archaeology scope.

**Concerns:**
- None observed at L2 depth.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §5. No KNOW-A* claim.

**Confidence:** **High** — small sub-package, file enumeration confirmed, acyclic status confirmed by oracle.

---

## C10. web/middleware/ — FastAPI middleware (acyclic — outside SCC)

**Path:** `src/elspeth/web/middleware/`
**Responsibility:** ASGI middleware — request-id propagation and composer-rate-limiting.
**File count, LOC:** 3 files, 257 LOC. `rate_limit.py`, `request_id.py`, `__init__.py`.
**Tests:** `tests/unit/web/middleware/` — 2 unit-test files.

**Internal coupling:** **NOT in SCC #4 — acyclic.** Inbound: `web/auth → web/middleware` weight 1 (intra_cluster_edges[49]). No outbound edges within cluster: middleware is leaf-like at the package level (it provides middleware classes; sub-packages consume them indirectly via app.py wiring, but the import graph collapses those references into the `web → web/middleware` edge that resolves at app.py).

**External coupling:** No direct outbound edges to other clusters.

**Patterns observed:**
- `request_id.py:RequestIdMiddleware(BaseHTTPMiddleware)` is true ASGI middleware (contrast with `web/auth/middleware.py` which is a FastAPI dependency, not ASGI middleware — see C3).
- `request_id.py:_is_safe_request_id(value)` and `_generate_request_id()` enforce a sanitised request-id contract before propagation — tier-3 boundary handling.
- `rate_limit.py:ComposerRateLimiter` is named for the composer specifically — rate-limiting is feature-scoped, not application-wide.

**Concerns:**
- The "ASGI middleware vs FastAPI dependency" distinction (between `web/middleware/` and `web/auth/middleware.py`) is a naming hazard for readers; would benefit from a clarifying note in middleware/__init__.py. Surface only.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §5. No KNOW-A* claim.

**Confidence:** **High** — small sub-package, file enumeration confirmed.

---

## C11. web/frontend/ — Out-of-scope (boundary record)

**Path:** `src/elspeth/web/frontend/`
**Status:** **OUT OF SCOPE** per Δ6 (L1 §3 — Python-lens archaeologist; ~13k LOC of TypeScript/React).
**Recorded contents (top-level only):** `dist/`, `node_modules/`, `src/`, `index.html`, `package.json`, `package-lock.json`, `react_effect_order.mjs`, `tsconfig.app.json`, `tsconfig.json`, `tsconfig.test.json`.
**Python-side reference:** [CITES KNOW-G7] — `web/app.py:create_app(...)` mounts `frontend/dist/` as a static-file route after API/WS routes. Per the KNOW-G7 invariant, this mount is order-sensitive (the static mount catches anything not matched by an API/WS route).
**No analysis performed on TSX, components, or build artefacts.** A frontend-aware archaeologist (e.g., the `lyra-site-designer` skillpack or a JS/TS-specialised codebase explorer) is the correct tool for this subtree per L1 §6.5.

**Confidence:** **N/A** (boundary record, not an analytical entry).

---

## Cross-cutting observations

These are *not* per-entry concerns — they apply across multiple entries:

- **The composer state machine is a cluster-internal central type.** Three independent consumers (composer_mcp transport, sessions persistence, execution validation) each pull from `web/composer/state.py` and `web/composer/tools.py`. F1's "transport over composer" framing is correct but understates the structural role; web/composer is the cluster's data backbone. *(Anchors §7.5 F1.)*
- **The 7-node SCC is FastAPI-app-factory-shaped.** `app.py` outwardly wires sub-packages; sub-packages reach back for shared types (WebSettings, run_sync_in_worker, paths). Both directions are intentional. *(Anchors §7.5 F4. SCC analysis section in 04-cluster-report.md elaborates.)*
- **The cluster has 0 inbound edges from other clusters.** It is consumed only by its console-script entry points (`elspeth-web`, `elspeth-composer`), not by any library code. This is the structural signature of an application-surface cluster. *(Sets cluster boundary discipline for synthesis.)*
- **Bug-hiding pattern findings (269 + 7) are pre-existing L2 debt.** Not blockers; surfaced in the cluster report's debt section and not propagated into per-entry text.

## L1 status updates (per Δ L2-3)

The following L1-pass open questions are closed by this catalog:

- **Q2 (`web/composer ↔ composer_mcp` relationship):** **Closed**. composer_mcp is a thin MCP transport over `web/composer/`'s state machine; one state machine, two transports (HTTP + MCP). [Confirmed by §7.5 F1; reinforced at symbol level in C1.]
- **Q3 (`mcp/ vs composer_mcp/` independence):** **Confirmed independent** at L1. The composer cluster does not analyse `mcp/`; F2 closed this question separately. [Reaffirmed; no new evidence.]
