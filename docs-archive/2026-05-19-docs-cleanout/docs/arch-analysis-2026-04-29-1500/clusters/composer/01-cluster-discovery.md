# 01 — Cluster Discovery (composer cluster: web/ + composer_mcp/)

This is an L2 cluster discovery pass — one level deeper than the L1 catalog (02-l1-subsystem-map.md §5 and §7) but capped at the sub-subsystem (immediate subdirectory or coherent file group) tier. Per Δ L2-3, files >1,500 LOC are FLAGGED as L3 deep-dive candidates and not summarised inline.

## 1. Scope and boundary

The composer cluster is two scope roots in the same architectural cluster, established by §7.5 F1 (`composer_mcp → web/composer` weight 12) and reinforced at the symbol level here:

```
composer_mcp/server.py imports:
  from elspeth.web.composer.state import CompositionState, PipelineMetadata
  from elspeth.web.composer.tools import (...)
  from elspeth.web.composer.yaml_generator import generate_yaml
  from elspeth.web.composer.redaction import redact_source_storage_path
  from elspeth.web.catalog.protocol import CatalogService
```

Five distinct module surfaces from `web/composer/` and one from `web/catalog/` are imported by `composer_mcp/server.py:1–40`. **The composer cluster has one composer state machine (in `web/composer/`) and two transports (HTTP via `web/`, MCP via `composer_mcp/`).** That answers F1 at the symbol level — composer_mcp/ is transport-only, not a peer composer.

The L1 §6 oracle (Phase 0) settled F2 separately: `mcp/` and `composer_mcp/` share zero import edges. `mcp/` is a different cluster and is not analysed here.

`web/frontend/` is recorded for boundary discipline only; no TSX analysed.

## 2. The cluster's internal structure

| Sub-package | Files (top-level Python) | LOC | Tests | In SCC #4? | Notes |
|---|---|---|---|---|---|
| `web/` (top-level files) | 7 | 1,072 | 4 root tests | YES (the package node "web") | `app.py` 652 LOC is the cycle's wiring root |
| `web/auth/` | 8 | 1,224 | 7 unit | YES | local auth + OIDC + middleware + protocol |
| `web/blobs/` | 6 | 1,952 | 5 unit | YES | blob upload/download with quota |
| `web/catalog/` | 5 | 407 | 3 unit | **NO (acyclic)** | catalog service exposing plugin metadata |
| `web/composer/` | 11 (12 with subpkg) | 8,189 (8,274 total) | 13 unit | YES | LLM-driven pipeline composer; **2 L3 deep-dive candidates inside** |
| `web/execution/` | 11 | 3,748 | 8 unit | YES | run-execution service (HTTP-side) |
| `web/middleware/` | 3 | 257 | 2 unit | **NO (acyclic)** | request-scope middleware |
| `web/secrets/` | 6 | 1,011 | 5 unit | YES | server-stored secrets |
| `web/sessions/` | 9 | 4,080 | 9 unit | YES | session/auth-token persistence |
| `composer_mcp/` | 3 | 824 | 2 unit | (separate scope root; LEAF) | MCP server transport over `web/composer/` state |
| `web/frontend/` | (TSX, out of scope) | ~13k TS/React | — | n/a | recorded only |

**Sub-package totals (Python only, in-scope):** ~22,058 LOC across 70 files for `web/` + 824 LOC across 3 files for `composer_mcp/`. Cluster total **~22,882 Python LOC**, plus the deferred ~13k LOC of `web/frontend/` TSX which is out of scope.

*LOC measurement note (per validator Note 4):* Per-sub-package LOC values in the table above come from `find <dir> -name "*.py" -exec cat {} + | wc -l`; the L1 catalog's web/-only figure (22,558 LOC, 02-l1-subsystem-map.md §5) was measured at L1 phase and may use a different counter (e.g., `cloc` or excluding empty `__init__.py`). The numbers are within 1% but do not reconcile to the byte; both are reported approximate. Synthesis pass should standardise on a single counter if exact arithmetic is needed.

## 3. Cycle structure (the 7-node SCC)

`[ORACLE: strongly_connected_components[4] = ['web', 'web/auth', 'web/blobs', 'web/composer', 'web/execution', 'web/secrets', 'web/sessions']]` — 7 nodes; entirely contained in the cluster (`temp/intra-cluster-edges.json:cluster_sccs[0].all_in_cluster=true`).

The cycle direction is **bidirectional** between the package root and its sub-packages:

- **Outward leg** (`web/` → sub-packages): `web/app.py` lines 1–60 import from every sub-package — `web/auth/{local,middleware,protocol,routes}`, `web/blobs/{routes,service}`, `web/catalog/routes`, `web/composer/...`, `web/execution/...`, `web/secrets/...`, `web/sessions/...`. This is the FastAPI app-factory wiring: `app.py` exists to *compose* the sub-packages into a single FastAPI app.
- **Inward leg** (sub-packages → `web/`): sub-packages import `from elspeth.web.config import WebSettings`, `from elspeth.web.async_workers import run_sync_in_worker`, `from elspeth.web.paths import ...`. These are shared infrastructure types that all sub-packages need (settings, async dispatch helpers, path constants).

Both directions are intentional. Removing one would either break wiring (no FastAPI app) or duplicate shared types into every sub-package. The cycle is the canonical FastAPI app-factory pattern realised at module scope; it is structurally load-bearing.

`web/catalog/` and `web/middleware/` participate in the wiring (`app.py` imports their routers/handlers) but **do not import back into top-level `web/`** at the package level the oracle measures, so the oracle correctly excludes them from the SCC.

## 4. External coupling (cited from `temp/intra-cluster-edges.json`)

**Inbound edges (other clusters → composer cluster):** **0**. *No other cluster imports anything from `web/` or `composer_mcp/`.* The composer cluster is a sink in the L3 import graph — it is consumed only by entry points (the `elspeth-web` and `elspeth-composer` console scripts), not by other library code.

**Outbound edges (composer cluster → other clusters):** **6 total**, summarised below; full records in `temp/intra-cluster-edges.json:cross_cluster_outbound_edges`.

| From | To | Weight | Conditional? | Sample site |
|---|---|---|---|---|
| `web` | `plugins/infrastructure` | 1 | no | `web/dependencies.py:48` |
| `web/catalog` | `plugins/infrastructure` | 3 | no | `web/catalog/service.py:8`, `:9` |
| `web/composer` | `plugins/infrastructure` | **22** | no | `web/composer/_semantic_validator.py:47–53` |
| `web/execution` | `.` (root `elspeth` package) | 3 | no | `web/execution/service.py:30,805`; `web/execution/validation.py:24` |
| `web/execution` | `plugins/infrastructure` | 4 | no | `web/execution/_semantic_helpers.py:66,67`; `web/execution/validation.py:30` |
| `web/execution` | `telemetry` | 1 | **yes** | `web/execution/service.py:781` |

Implicit cross-cluster facts (deferred to synthesis):
- The cluster does NOT directly import from `engine/` at the module-collapse granularity the oracle measures — it routes through `plugins/infrastructure/` instead. This is consistent with the L1 catalog's claim that engine instantiates plugins via the registry; web reads the registry through `plugins/infrastructure/` to surface plugin metadata and validate composer outputs.
- The `web/execution → .` edge (root `elspeth` package) is unusual at this granularity. The sample sites are in `service.py` (the run-execution path) and `validation.py`. The likely cause is `from elspeth import <something at package root>` — possibly a re-exported public symbol. This is a single observation; the catalog flags it without diagnosing.
- The conditional `web/execution → telemetry` edge (weight 1) is the cluster's only conditionally-imported cross-cluster dependency. Conditional imports typically indicate runtime-feature gating (telemetry available only when configured).

## 5. Layer compliance

`scripts/cicd/enforce_tier_model.py check` was run for both scope roots:

- `web/`: 269 findings across the bug-hiding-pattern rules (R1 ×137, R2 ×3, R5 ×63, R6 ×53, R7 ×4, R8 ×2, R9 ×7) — `isinstance` checks, silent `except`, and related anti-patterns. **Zero layer (L1-rule) findings outside the documented `engine/*` allowlist exemption.**
- `composer_mcp/`: 7 findings (R1 ×4, R5 ×3), all bug-hiding-pattern rules. **Zero layer findings.**

The cluster is layer-conformant; bug-hiding pattern findings are L2 debt candidates, not architecture findings, and are surfaced in the cluster report's debt section without being propagated into per-entry catalog text.

`dump-edges` byte-equality vs the L1 oracle holds (77/77 edges, 33/33 nodes, 5/5 SCCs).

## 6. Out-of-scope material recorded for completeness

- `src/elspeth/web/frontend/` — ~13k LOC of TypeScript/React, plus `node_modules/`, `dist/`, `package.json`, `package-lock.json`, `tsconfig.{app,test}.json`, `index.html`, `react_effect_order.mjs`. Not analysed. The Python-side reference is the FastAPI static-file mount in `app.py` (cited but not enumerated).
- `tests/integration/composer_mcp/` — directory ABSENT. Recorded in cluster report as a Δ L2-5 test-coverage debt candidate.

[CITES KNOW-G6, KNOW-G7] for the staging deployment context (uvicorn factory `elspeth.web.app:create_app`, frontend served from `web/frontend/dist/`). [CITES KNOW-G9] for the composer-specific provider-error opt-in switch.

## 7. What this discovery establishes

| Question entering this pass | Answer | Confidence |
|---|---|---|
| Is composer_mcp/ a sibling or nested under web/composer/? | Nested *behaviourally*; structured as sibling for transport isolation. composer_mcp/ imports web.composer.{state,tools,...} — single state machine, two transports. | High (symbol-level evidence). |
| Is the 7-node SCC essential or accidental? | Essential. Bidirectional FastAPI app-factory pattern: app.py wires sub-packages, sub-packages reach back for shared types. Removing either direction breaks wiring or duplicates types. | High (cycle direction inspected; KNOW-G6 confirms `app.py:create_app` is the deployed entry). |
| Are web/catalog and web/middleware acyclic siblings within scope but outside the SCC? | Yes — confirmed by oracle and by inspection. They participate in the wiring but do not import back. | High. |

The cluster is unusually self-contained: 0 inbound edges, 6 outbound (5 to `plugins/infrastructure`, 1 conditional to `telemetry`, 1 to root `elspeth`). The headline structural facts are the F1-confirmed transport relationship and the F4 SCC's load-bearing-ness.
