# 04 — Cluster Report (composer cluster: web/ + composer_mcp/)

This is the synthesis pass for the L2 #2 composer cluster. It does not re-run the L1 catalog, does not contradict §7.5 amendments F1–F5, and does not propose decompositions for SCC #4 (Δ L2-7 — surface the question, defer the prescription).

L1 back-references cited at least once each per the specialisation block: 02-l1-subsystem-map.md §5 (web), §7 (composer_mcp); 04-l1-summary.md §7 Priority 2; §7.5 F1, F2, F4; "Closed L1 open questions" Q2 (closed by F1) and Q3 (closed by F2); KNOW-G6, KNOW-G7, KNOW-G9.

## 1. Cluster shape

The composer cluster is two scope roots in one architectural cluster: `src/elspeth/web/` (L3, COMPOSITE, 72 files / 22,558 Python LOC; `web/frontend/` excluded per Δ6) and `src/elspeth/composer_mcp/` (L3, LEAF, 3 files / 824 LOC). Total in-scope: ~22,882 Python LOC across 75 files.

The cluster's structural signature, from the filtered import graph (`temp/intra-cluster-edges.json`):
- **35 intra-cluster edges** across the 9 in-scope sub-package nodes (10 if you count `web/composer/skills/`).
- **0 inbound edges** from any other cluster — the composer cluster is a sink in the L3 import graph.
- **6 outbound edges** to other clusters: 4 to `plugins/infrastructure` (aggregate weight 30, dominated by `web/composer → plugins/infrastructure` weight 22), 1 to `telemetry` (conditional, weight 1), 1 to root `elspeth` (weight 3).
- **1 strongly-connected component** entirely contained in the cluster: SCC #4 `['web', 'web/auth', 'web/blobs', 'web/composer', 'web/execution', 'web/secrets', 'web/sessions']`, 7 nodes.

The cluster is consumed only by its console-script entry points (`elspeth-web` via `web.app:create_app`, `elspeth-composer` via `composer_mcp:main`), not by any library code in other clusters. This is the structural signature of an application-surface cluster.

## 2. SCC analysis (Δ L2-7 — mandatory section)

### 2.1. Cycle composition

`[ORACLE: strongly_connected_components[4] = ['web', 'web/auth', 'web/blobs', 'web/composer', 'web/execution', 'web/secrets', 'web/sessions']; size 7; oracle path: temp/l3-import-graph.json:strongly_connected_components[4]]`

Per Δ L2-7, each member has a dedicated catalog entry that cites the cycle explicitly (entries C2–C8 in `02-cluster-catalog.md`). The two acyclic siblings within the same scope root (`web/catalog/`, `web/middleware/`) are analysed in C9 and C10 — they participate in the wiring (`app.py` imports their routers/middleware classes) but do not import back into the package root or other sub-packages.

### 2.2. Why the cycle exists

The cycle is FastAPI-app-factory-shaped, with two independent direction-of-flow rationales:

- **Outward leg (`web/` package → sub-packages).** `web/app.py:create_app(...)` is the deployment entry per [CITES KNOW-G6]; its job is to compose every sub-package's router/dependency into a single FastAPI app. To do so, `app.py:1–60` imports `web.auth.{local,middleware,protocol,routes}`, `web.blobs.{routes,service}`, `web.catalog.routes`, `web.composer.*`, `web.execution.*`, `web.secrets.*`, `web.sessions.*`. This is necessary at the wiring layer.
- **Inward leg (sub-packages → `web/` package).** Sub-packages reach back for shared types: `from elspeth.web.config import WebSettings` (in every sub-package — sub-packages need configuration), `from elspeth.web.async_workers import run_sync_in_worker` (the async-worker dispatch helper, in sub-packages that mix sync work with async routes), `from elspeth.web.paths import ...` (path constants).

The cycle exists because `web/` is simultaneously the *wiring* layer (which must know about its sub-packages) and the *shared-infrastructure* layer (which sub-packages must know about). Both directions are intentional; neither is artefactual.

### 2.3. What it would take to break the cycle

The cycle could be broken in principle by separating the two roles:

- Extract `WebSettings`, `run_sync_in_worker`, and path constants out of `web/` into a non-cyclic spine (e.g., `web/_infrastructure/` or a new `web/contracts/` directory). Sub-packages would then import from the spine instead of the package root.
- Or invert the wiring: have sub-packages register themselves with `app.py` via a registry mechanism rather than `app.py` importing them concretely.

Both approaches would require non-trivial refactoring. The first reshuffles ~150 LOC of shared types but leaves the wiring direction intact; the second eliminates the outward leg but requires plug-style discovery, which is heavier infrastructure than the current direct imports.

### 2.4. Is the cycle load-bearing?

Yes, in its current form. The evidence:

- **`app.py` is the deployed entry point.** [CITES KNOW-G6] — `uvicorn elspeth.web.app:create_app --factory`. The wiring direction is operationally required.
- **Sub-packages tested independently.** Each sub-package has its own `tests/unit/web/<name>/` directory with 2–13 test files. Tests do not need the cycle to exercise sub-package logic — they import sub-package code directly. The cycle is a *deployment-path* property, not a *test-path* property.
- **No test currently asserts the cycle.** The cycle is a structural property the oracle measures, not an explicit invariant. The catalog records this as L2 debt: an explicit test that asserts "every sub-package can be imported standalone" would lock in the inward-leg-only direction; a separate test asserting "app.py imports every sub-package's router" would lock in the outward leg.

### 2.5. Architecture-pack questions (deferred per Δ L2-7)

Surfaced for the architecture-pack pass, *not* prescribed here:

- Should `web/` extract its shared infrastructure (`WebSettings`, `run_sync_in_worker`, path constants) into a non-cyclic spine to break the inward leg?
- Should `app.py` invert the wiring direction (sub-package self-registration vs concrete import)?
- Is the cluster boundary (`web/` + `composer_mcp/`) the right scope, or should `composer_mcp/` be relocated to `web/composer_mcp/` to reflect its transport-only role?

The catalog has no opinion on these; the archaeology pass surfaces the question.

## 3. Layer compliance summary

`scripts/cicd/enforce_tier_model.py check` was run for both scope roots:
- `web/`: 269 findings, all on bug-hiding-pattern rules (R1 ×137, R2 ×3, R5 ×63, R6 ×53, R7 ×4, R8 ×2, R9 ×7). Zero L1 (layer-rule) violations outside the documented `engine/*` allowlist exemption (which is unrelated to this cluster).
- `composer_mcp/`: 7 findings, all bug-hiding-pattern rules (R1 ×4, R5 ×3). Zero layer violations.

`dump-edges` byte-equality vs the L1 oracle: 77/77 edges, 33/33 nodes, 5/5 SCCs match. The cluster's intra-cluster filter (35 edges) is provably correct.

The cluster's layer compliance is **clean for layer rules**; the 276 bug-hiding-pattern findings are surfaced as L2 debt candidates, not architecture findings.

## 4. Debt candidates (L2-flagged)

Recorded for downstream synthesis and the architecture-pack pass:

1. **Bug-hiding pattern density.** 269 findings in `web/` and 7 in `composer_mcp/` on R1/R5/R6/R9 rules. The headline sites cited by the script include `app.py:128 except (SQLAlchemyError, OSError) as cleanup_exc:` and `web/composer/tools.py` (R1, R5, R9 — explicitly tagged as "Tier 3 boundary — LLM tool call arguments are external data"). Some findings are documented allowlist candidates; others are genuine debt.
2. **`tests/integration/composer_mcp/` absent.** The MCP transport has unit tests but no integration test exercising an MCP tool round-trip. End-to-end coverage gap (Δ L2-5 debt-flag).
3. **`web/composer/` is composite at L2 depth.** 11+1 files, 8,274 LOC, with 67% concentrated in `tools.py` (3,804) + `state.py` (1,710). Per the L1 Δ4 heuristic the sub-package qualifies as composite at L2 depth — flagged as "L3 candidate: composite at L2 depth" and not recursed.
4. **`web/sessions/` and `web/execution/` density.** 4,080 and 3,748 LOC respectively — second- and third-densest sub-packages. Below the L3 deep-dive LOC threshold, but high concentration; verifying their internal structure is L3 scope.
5. **The `web/execution → .` (root `elspeth`) edge.** Weight 3, sample sites `service.py:30,805` and `validation.py:24`. Unusual at this granularity; cause is not diagnosed at L2 depth (would need file inspection at the cited lines). Surface for synthesis.
6. **Naming hazard: "ASGI middleware" vs "FastAPI dependency".** `web/middleware/` is true ASGI middleware; `web/auth/middleware.py` is a FastAPI dependency function. Both use the word "middleware" but are different abstractions. Documentation-level concern; surface only.
7. **No KNOW-A* coverage of `web/` or `composer_mcp/`.** The institutional-knowledge map predates web-UI maturity; per-sub-package responsibility claims in this catalog are *building* knowledge rather than corroborating it. Listed in 04-l1-summary.md §6.5; reaffirmed here.

## 5. Highest-confidence claims (top 3 — for stitched report)

1. **The composer cluster has one composer state machine and three internal consumers.** `web/composer/state.py` (1,710 LOC) and `tools.py` (3,804 LOC) own the pipeline-composition state and tool surface. This state is consumed by `composer_mcp/` (transport, weight 12), `web/sessions/` (persistence, weight 15), and `web/execution/` (validation, weight 9). F1 stands at the symbol level (`composer_mcp/server.py:1–40` imports the state types directly); F1's "thin transport" framing is correct but understates the structural role — `web/composer/` is the cluster's data backbone, not just an MCP target.
2. **The 7-node SCC is the FastAPI app-factory pattern made structural.** `web/app.py:create_app(...)` outwardly imports every sub-package's router (the wiring leg); sub-packages reach back via `from elspeth.web.config import WebSettings` and `run_sync_in_worker` (the shared-infrastructure leg). Both directions are intentional. The cycle is load-bearing in its current form, and decomposition is a non-trivial refactoring decision left to the architecture-pack pass.
3. **The cluster has 0 inbound edges from any other cluster.** Confirmed by `temp/intra-cluster-edges.json:cross_cluster_inbound_edges = []`. The composer cluster is consumed only by its two console-script entry points (`elspeth-web`, `elspeth-composer`), not by library code elsewhere. This is the structural signature of an application-surface cluster, and it tightens what the post-L2 synthesis pass can plausibly say about cluster-level coupling: there are no surprise back-references to defer.

## 6. Highest-uncertainty questions (top 3 — agenda for post-L2 synthesis)

1. **What is the `web/execution → .` (root `elspeth` package) edge importing?** Weight 3, sample sites `web/execution/service.py:30,805` and `validation.py:24`. The catalog cannot diagnose this without file inspection at those lines. If it's a re-export of a public symbol from `elspeth/__init__.py`, that's benign. If it's something else (e.g., a deferred-import hack to bypass an explicit cluster dependency), that's a different finding.
2. **How does the composer cluster's secrets surface compose with cross-cluster secret handling?** `web/secrets/` has zero outbound edges to other clusters at the package-collapse granularity, yet composer/execution rely on LLM-provider credentials that are presumably loaded through the secrets surface and then handed to plugin code. Is the credential flow happening via `WebSettings` injection at request time (not visible to the import graph)? L3 inspection territory.
3. **Why is `web/sessions → web/composer` weight 15 (joint-heaviest intra-cluster edge)?** The catalog interprets this as "sessions persists composer drafts" based on the file names (`engine.py`, `service.py:_assert_state_in_session`), but the symbol-level evidence has not been inspected. Confirming the data-flow direction (sessions reads/writes composer state types vs composer reads session metadata) is L3 scope.

## 7. Cross-cluster observations for synthesis (deferred from Δ L2-4)

Recorded here verbatim for the post-L2 stitching pass; not analysed in this cluster:

- **The cluster does not import directly from `engine/` at the package-collapse granularity.** It routes through `plugins/infrastructure/` (3 of the 6 outbound edges target this) and through `plugins/infrastructure/`-routed plugin metadata. This is consistent with the L1 catalog's "engine instantiates plugins via the registry" claim, but the synthesis pass should confirm the same pattern holds in the engine and plugins clusters' L2 catalogs.
- **The conditional outbound edge `web/execution → telemetry` (weight 1) is the cluster's only conditional cross-cluster dependency.** Whether telemetry-conditional imports are a cluster-specific pattern or a project-wide pattern is a synthesis question; this catalog records yes for this cluster only.
- **`mcp/` and `composer_mcp/` are confirmed independent siblings at the import level (F2).** This catalog reaffirms the L1 "do not merge" guidance for `mcp/` but does not analyse `mcp/`. The synthesis pass should validate that `mcp/`'s own L2 pass (if undertaken) does not contradict this.
- **The cluster's two console-script entry points are the only inbound consumers.** No other cluster imports anything from `web/` or `composer_mcp/`. The synthesis pass can therefore treat the composer cluster as a *terminal* cluster in the import-direction sense — it consumes upstream clusters but is not consumed by them.

## 8. L1-pass status updates

- **Q2 (web/composer ↔ composer_mcp):** *Closed.* Confirmed at symbol level: composer_mcp imports from `web.composer.{state, tools, yaml_generator, redaction, protocol}` and from `web.catalog.protocol`. There is no parallel state machine in `composer_mcp/`. *(Closes L1 open question Q2; aligns with §7.5 F1.)*
- **Q3 (mcp/ vs composer_mcp/):** *Confirmed independent.* `mcp/` has zero import edges into the composer cluster scope (and vice versa). *(L1's §7.5 F2 closure stands; reaffirmed here.)*
- **KNOW-A70 status:** Not in scope. KNOW-A70 is the engine/orchestrator/processor/coalesce concentration flag and was the engine cluster's calibration anchor (L2 #1, completed previously). Does not apply to the composer cluster.

## 9. Outputs produced by this pass

| File | Status |
|---|---|
| `00-cluster-coordination.md` | written |
| `01-cluster-discovery.md` | written |
| `02-cluster-catalog.md` | written (11 entries: composer_mcp + 7 SCC + 2 acyclic + 1 frontend record) |
| `03-cluster-diagrams.md` | written (Container + Component) |
| `04-cluster-report.md` | this file |
| `temp/intra-cluster-edges.json` | written (35 intra / 0 inbound / 6 outbound / 1 SCC) |
| `temp/intra-cluster-edges-rederived-web.json` | NOT produced separately — Δ L2-6 byte-equality verified against the full L1 oracle (77/77 edges) instead, which is the truer comparison and which the cluster's filter is mathematically derived from |
| `temp/layer-check-web.txt` | written (269 findings, 0 layer-rule violations) |
| `temp/layer-check-composer_mcp.txt` | written (7 findings, 0 layer-rule violations) |
| `temp/validation-l2.md` | spawned via `analysis-validator` subagent (Δ L2-8) — pending or complete depending on synchronous status |

The post-L2 synthesis pass owns top-level updates (`00-coordination.md`, `04-l1-summary.md`); per Δ L2-4 this cluster pass does not modify them.
