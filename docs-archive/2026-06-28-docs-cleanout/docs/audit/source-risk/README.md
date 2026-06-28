# Source Bug-Risk Sweep

Evidence-first source sweep for `src/elspeth`, parallel to the test-suite audit
in `docs/audit/test-suite/`. The rating estimates where architecture smell,
latent bugs, correctness drift, or high-leverage improvement opportunities are
most likely to exist. It is not a finding list; it is the triage map for a
future folder-by-folder deep dive.

## Rating Scale

| Rating | Priority | Meaning |
|---|---:|---|
| 5 | P1 | Highest-risk source surface: audit/legal correctness, data trust boundary, large state machine, or known active debt |
| 4 | P1 | High-risk source surface: complex cross-module contracts, external systems, persistence, or UI/runtime parity |
| 3 | P2 | Medium-risk surface: meaningful defects plausible, but smaller blast radius or stronger local boundaries |
| 2 | P3 | Low-risk source surface: mostly harness/config/support or already narrow leaf behavior |
| 1 | P3 | Very low-risk source surface: tiny stable declarations or packaging glue |

## Method

- Inventory source with `git ls-files src/elspeth`, excluding generated/vendor
  directories such as `node_modules`, `.e2e-data`, `dist`, and caches.
- Chunk by cohesive runtime ownership, not by arbitrary line-count slices.
- Prefer risk signals that match ELSPETH standards: trust-tier boundaries,
  audit primacy, config-contract mapping, closed-list governance, state
  persistence, external calls, and production-code-path divergence.
- Use comments as evidence where they are explicit load-bearing guidance, e.g.
  closed lists, known gaps, workaround deletion criteria, and mechanical
  contract notes.

## Sweep List

| ID | Source folder / scope | Files | LOC | Risk | Priority | Why this belongs in the sweep |
|---|---|---:|---:|---:|---:|---|
| S-CORE-LANDSCAPE | `src/elspeth/core/landscape/` | 20 | 10,293 | 5 | P1 | Audit database is the legal record. Schema and repository code carry composite FK ownership, checkpoint compatibility, query/export, and lineage invariants; a small drift can invalidate run attribution. |
| S-ENGINE-CORE | `src/elspeth/engine/` excluding `executors/` | 22 | 13,121 | 5 | P1 | Orchestrator and processor own run lifecycle, resume, checkpointing, terminal outcomes, token traversal, and row accounting. `orchestrator/core.py` has a documented resume progress gap, making it a prime deep-dive target. |
| S-WEB-COMPOSER-CORE | `src/elspeth/web/composer/` excluding `guided/` | 28 | 27,215 | 5 | P1 | Largest Python surface in the repo. It mixes LLM calls, tool schemas, redaction, audit sidecars, prompt/skill drift, semantic validation, and session mutation; comments explicitly forbid config drift without ADR and mention follow-up migration debt. |
| S-WEB-SESSIONS | `src/elspeth/web/sessions/` | 23 | 17,727 | 5 | P1 | Session persistence, chat history, run records, locks, event tables, route helpers, and proposal state live here. The code has process-wide lock invariants, field-overloading debt, and multiple closed-list SQL governance boundaries. |
| S-PLUGINS-EXTERNAL-TRANSFORMS | `src/elspeth/plugins/transforms/{llm,rag,azure}/`, `web_scrape*`, `safety_utils.py` | 28 | 7,777 | 5 | P1 | External-call transform boundary: LLM, HTTP/web scrape, retrieval, and Azure safety responses are Tier 3 inside otherwise Tier 2 row processing. Immediate validation, retry classification, SSRF controls, and audit-call recording are likely bug seams. |
| S-CONTRACTS-CORE | `src/elspeth/contracts/` excluding `config/` | 67 | 18,503 | 4 | P1 | Core wire/data contracts, registries, errors, row/result types, audit evidence, closed enums, hashing, schema contracts, and runtime manifests. Drift here fans out through engine, plugins, web, MCP, and frontend DTOs. |
| S-CONTRACTS-CONFIG | `src/elspeth/contracts/config/`, `src/elspeth/core/config.py`, `src/elspeth/core/dependency_config.py`, `src/elspeth/web/config.py` | 8 | 4,127 | 4 | P1 | Settings-to-runtime mapping is an established defect class. Any orphaned field, default mismatch, or missing protocol member silently changes runtime behavior after YAML validation appears to pass. |
| S-CORE-DAG-CONTRACTS | `src/elspeth/core/dag/` plus core templates, expression parser, canonical JSON, operations, events, identifiers | 11 | 5,399 | 4 | P1 | DAG validation, schema propagation, template extraction, expression parsing, canonicalization, and graph topology are central correctness gates. Bugs here can make invalid pipelines look valid before execution. |
| S-ENGINE-EXECUTORS | `src/elspeth/engine/executors/` | 16 | 5,625 | 4 | P1 | Executes transform/gate/sink/aggregation contracts and declaration dispatch. It enforces field collisions, required fields, pass-through behavior, and sink diversion semantics, so local gaps become row-level audit defects. |
| S-PLUGINS-INFRA | `src/elspeth/plugins/infrastructure/` | 43 | 11,432 | 4 | P1 | Plugin discovery, manager, config base, schema factory, clients, batching, pooling, URL validation, display headers, and telemetry. Dynamic folder scanning plus client boundaries make it a high-payoff review area. |
| S-PLUGINS-IO | `src/elspeth/plugins/{sources,sinks}/` | 15 | 7,417 | 4 | P1 | Source and sink plugins are the Tier 3 ingress/egress boundary. Header normalization, output schemas, path handling, cloud/database clients, fixed-mode schema checks, and audit-failure classification all sit here. |
| S-PLUGINS-BATCH-TRANSFORMS | `src/elspeth/plugins/transforms/batch_*.py`, `report_assemble.py` | 13 | 5,261 | 4 | P1 | Batch/aggregation transforms have deaggregation, grouping, statistics, thresholds, and report assembly behavior. They are likely to hide row-count, field-propagation, and audit-summary drift. |
| S-WEB-EXECUTION | `src/elspeth/web/execution/` | 18 | 7,377 | 4 | P1 | Web execution is the bridge from composed YAML to runtime services and live status. It owns validation, runtime preflight, diagnostics, progress, output finalization, and terminal-status projection. |
| S-WEB-COMPOSER-GUIDED | `src/elspeth/web/composer/guided/` | 18 | 3,731 | 4 | P1 | Guided state machine, recipe matching, emitters, audit events, and step prompts are compact but stateful. Closed-list step/turn comments and prompt/tool coupling make it likely to drift from the freeform composer. |
| S-WEB-DATA-SURFACES | `src/elspeth/web/{blobs,secrets,shareable_reviews,preferences,catalog,audit_readiness}/` plus `paths.py`, `validation.py` | 35 | 7,704 | 4 | P1 | Blob storage, secret refs, signed share links, preferences, catalog lowering, and audit readiness are all user-facing trust or persistence surfaces. Mistakes here can leak data, misrepresent plugin safety, or break runtime handoff. |
| S-FRONTEND-API-STATE | `src/elspeth/web/frontend/src/{api,stores,hooks,types,utils,contexts,lib}/` | 69 | 18,040 | 4 | P1 | Client DTOs and state stores must stay aligned with Python schemas and live WebSocket/API semantics. Session/execution stores are large enough to hide stale status projection and cancellation/recovery bugs. |
| S-FRONTEND-COMPONENTS | `src/elspeth/web/frontend/src/components/` | 163 | 37,531 | 4 | P1 | Largest frontend surface. Chat, guided turns, inspector graph, audit readiness, recovery, blobs, settings, tutorial, catalog, and execution panels all encode UX/runtime contracts that can drift from backend behavior. |
| S-CORE-STATE-SECURITY | `src/elspeth/core/{checkpoint,security,rate_limit,retention}/` plus payload store, secrets, logging | 18 | 3,951 | 3 | P2 | Important but smaller scoped primitives: checkpoint serialization, secret loading, SSRF/IP validation, payload purge, rate limits, logging setup. Risk is meaningful, but modules are more leaf-shaped than engine/web orchestrators. |
| S-PLUGINS-LOCAL-TRANSFORMS | Non-batch local transforms in `src/elspeth/plugins/transforms/` | 11 | 2,696 | 3 | P2 | Field mapper, JSON/line explode, keyword filter, truncate, type coercion, value transform, and passthrough are local but shape-changing. Risk concentrates around field contracts, collision handling, and over-defensive transform behavior. |
| S-WEB-APP-CORE | `src/elspeth/web/*.py`, `auth/`, `middleware/` | 18 | 3,330 | 3 | P2 | App wiring, dependencies, auth providers, request IDs, and rate limiting are important but smaller. Main risk is middleware/auth ordering, config injection, and staging/frontend static serving interactions. |
| S-MCP-TUI-TELEMETRY | `src/elspeth/{mcp,tui,telemetry,composer_mcp}/` | 36 | 9,797 | 3 | P2 | Operator surfaces and observability. MCP has a documented parent-death workaround; telemetry exporters have many best-effort external paths; TUI reads audit records. Review for drift and failure classification. |
| S-ROOT-CLI | Root package files and `src/elspeth/testing/` | 6 | 3,927 | 3 | P2 | CLI is a broad operator entrypoint and `testing/` helpers can encode false production paths. Risk is medium because logic mostly delegates, but bad helper patterns can seed many weak tests. |
| S-FRONTEND-APP-SHELL | Frontend app shell, CSS/tokens/config/public/build config | 22 | 9,438 | 3 | P2 | `App.css` is large and load-bearing for guided widgets and accessibility states. Risk is mostly UX/a11y/theming drift rather than data corruption. |
| S-FRONTEND-TEST-E2E | Co-located frontend test setup and Playwright e2e under `src/elspeth/web/frontend/` | 31 | 4,664 | 2 | P3 | This is source-adjacent test/support code, not production runtime. Include for completeness because it can hide staging assumptions, auth setup drift, and false confidence, but it belongs behind production folders. |

## Recommended Review Order

1. Start with the rating-5 P1 surfaces: `core/landscape`, `engine`, `web/composer`, `web/sessions`, and external transforms.
2. Then review rating-4 P1 contract and boundary surfaces: contracts, config contracts, DAG, executors, plugin infrastructure, plugin IO, batch transforms, web execution, guided composer, web data surfaces, and frontend API/state/components.
3. Finish with P2/P3 surfaces unless an active incident points at them: core state/security, local transforms, web app core, MCP/TUI/telemetry, root CLI/testing, frontend shell, and frontend e2e support.

## Tracker Mapping

Created tracker structure:

- Epic: `elspeth-4088c4c604` — `Bug risk epic — src/elspeth architecture smell and latent defect review`
- One child `task` per row in the sweep list.
- Priority mapping: ratings 5 and 4 -> P1, rating 3 -> P2, ratings 2 and 1 -> P3.
- Each task should cite this document and carry labels `audit:source-risk-sweep` and `risk-rating:<n>`.

| Sweep ID | Filigree task | Priority |
|---|---|---:|
| S-CORE-LANDSCAPE | `elspeth-4b6a1f4dcb` | P1 |
| S-ENGINE-CORE | `elspeth-097b565746` | P1 |
| S-WEB-COMPOSER-CORE | `elspeth-854cc963c2` | P1 |
| S-WEB-SESSIONS | `elspeth-00ab664404` | P1 |
| S-PLUGINS-EXTERNAL-TRANSFORMS | `elspeth-bf322e3799` | P1 |
| S-CONTRACTS-CORE | `elspeth-0ed307cc64` | P1 |
| S-CONTRACTS-CONFIG | `elspeth-ca0bc179f5` | P1 |
| S-CORE-DAG-CONTRACTS | `elspeth-e5dcce584b` | P1 |
| S-ENGINE-EXECUTORS | `elspeth-1d312ad9cc` | P1 |
| S-PLUGINS-INFRA | `elspeth-66f5b70e17` | P1 |
| S-PLUGINS-IO | `elspeth-b8507e2f04` | P1 |
| S-PLUGINS-BATCH-TRANSFORMS | `elspeth-7630a3642e` | P1 |
| S-WEB-EXECUTION | `elspeth-209a875e89` | P1 |
| S-WEB-COMPOSER-GUIDED | `elspeth-f9e32ec973` | P1 |
| S-WEB-DATA-SURFACES | `elspeth-9aeadc95e9` | P1 |
| S-FRONTEND-API-STATE | `elspeth-a1bda77c7e` | P1 |
| S-FRONTEND-COMPONENTS | `elspeth-7efbb06f86` | P1 |
| S-CORE-STATE-SECURITY | `elspeth-2edc392912` | P2 |
| S-PLUGINS-LOCAL-TRANSFORMS | `elspeth-363a699247` | P2 |
| S-WEB-APP-CORE | `elspeth-16ee7676a3` | P2 |
| S-MCP-TUI-TELEMETRY | `elspeth-6f77e4eec0` | P2 |
| S-ROOT-CLI | `elspeth-5b920be58a` | P2 |
| S-FRONTEND-APP-SHELL | `elspeth-cf691873fb` | P2 |
| S-FRONTEND-TEST-E2E | `elspeth-2a6a24407d` | P3 |

## Recursive Inventory

The next pass expands the sweep to every tracked folder and file under
`src/elspeth`:

- Human summary: [recursive-summary.md](recursive-summary.md)
- Complete CSV inventory: [recursive-inventory.csv](recursive-inventory.csv)
- Recursive folder task map:
  [recursive-folder-tasks.csv](recursive-folder-tasks.csv)
- First populated file-task tree:
  [tree-S-CORE-LANDSCAPE.md](tree-S-CORE-LANDSCAPE.md)
- First populated file-task map:
  [recursive-file-tasks-S-CORE-LANDSCAPE.csv](recursive-file-tasks-S-CORE-LANDSCAPE.csv)

Coverage as of the recursive pass:

- 88 tracked-source folders rated
- 743 tracked files rated
- 831 total folder/file rows
- 87 recursive folder review tasks created in Filigree, excluding only the
  root `src/elspeth/` rollup because the epic already owns that boundary
- 20 file review tasks created under the `S-CORE-LANDSCAPE` branch

The CSV is the authoritative complete list. The summary highlights folder
rollups and the highest-risk files so the next reviewer can choose work without
opening a giant table first. Filigree folder tasks are prioritised from the
recursive rating: ratings 5 and 4 are P1, rating 3 is P2, and ratings 2 and 1
are P3. File tasks use the same priority mapping and sit under the most specific
folder task for that branch.
