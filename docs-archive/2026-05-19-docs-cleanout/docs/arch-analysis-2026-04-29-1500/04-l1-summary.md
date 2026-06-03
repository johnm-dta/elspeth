# 04 — L1 Summary, Deferrals, and L2 Dispatch Queue

## Executive summary

ELSPETH is a domain-agnostic auditable Sense/Decide/Act pipeline framework, ~121,392 LOC of production Python distributed across **11 verified top-level subsystems** under `src/elspeth/`. The codebase enforces a strict 4-layer model (L0 contracts → L1 core → L2 engine → L3 application surfaces) via `scripts/cicd/enforce_tier_model.py`, which ran clean for this analysis — every cross-layer dependency in the codebase is layer-conformant at scan time. Five subsystems are COMPOSITE per the Δ4 heuristic (`plugins/`, `web/`, `core/`, `engine/`, `contracts/`) and account for ~89% of LOC; six are LEAF (`mcp/`, `composer_mcp/`, `telemetry/`, `tui/`, `testing/`, `cli` root files). Twelve individual files exceed the 1,500-LOC L2-deep-dive threshold and are flagged but unread. **The L1 pass is complete and produces a prioritised L2 dispatch queue (§7) below; deep analysis of any composite is deferred.**

## 1. What this L1 pass verified

| Fact | Source |
|------|--------|
| Subsystem inventory: 11 (not 12) under `src/elspeth/` | `find src/elspeth -maxdepth 1` + verified counts in `00-coordination.md` |
| Total production Python LOC: 121,392 | `find … -print0 \| xargs -0 cat \| wc -l` per subsystem |
| Layer model is CI-enforced and currently clean | `temp/tier-model-oracle.txt`; `enforce_tier_model.py:190–250`, exit 0 |
| 5 COMPOSITE / 6 LEAF classification matches Δ4 expectation | `02-l1-subsystem-map.md` (Δ4 heuristic applied per entry) |
| Layer assignments: `contracts`=L0, `core`=L1, `engine`=L2, all others=L3 | Path-based table in `enforce_tier_model.py:237–248` |
| Cross-layer dependency graph at L1 granularity | Deterministic from layer assignments + clean enforcer (see `03-l1-context-diagram.md` §2) |

`[DIVERGES FROM]` the scope-override prose count of "12": the explicit Δ4 expected-classification list itself enumerates only 11, and the verified directory listing confirms 11. Treated as a prose typo; recorded in `00-coordination.md` and `02-l1-subsystem-map.md` headers.

## 2. The 11 subsystems at a glance

| # | Subsystem | Layer | Class | Files | LOC | Role (one phrase) |
|--:|-----------|-------|-------|------:|----:|-------------------|
| 1 | `contracts/` | L0 | COMPOSITE | 63 | 17,403 | Shared types, protocols, freeze primitives, hashing, security primitives |
| 2 | `core/` | L1 | COMPOSITE | 49 | 20,791 | Landscape audit DB, DAG, config, canonical JSON, payload store, retention, rate limit, security, expression parser |
| 3 | `engine/` | L2 | COMPOSITE | 36 | 17,425 | Orchestrator, RowProcessor, executors, retry, artifact pipeline, span factory, triggers |
| 4 | `plugins/` | L3 | COMPOSITE | 98 | 30,399 | System-owned Sources/Transforms/Sinks + audited HTTP/LLM clients + hookspecs |
| 5 | `web/` | L3 | COMPOSITE | 72 | 22,558 | FastAPI server: composer, execution, catalog, auth, sessions, secrets, blobs, middleware (frontend SPA out of scope) |
| 6 | `mcp/` | L3 | LEAF | 9 | 4,114 | Read-only Landscape audit-DB analyser MCP server (`elspeth-mcp`) |
| 7 | `composer_mcp/` | L3 | LEAF | 3 | 824 | Stateful pipeline-construction MCP server (`elspeth-composer`) |
| 8 | `telemetry/` | L3 | LEAF | 14 | 2,884 | Operational telemetry pipeline (post-Landscape, audit-primacy compliant) |
| 9 | `tui/` | L3 | LEAF | 9 | 1,175 | Textual TUI for `elspeth explain` audit-trail traversal |
| 10 | `testing/` | L3 | LEAF | 2 | 877 | pytest plugin (`elspeth-xdist-auto`) — NOT to be confused with `tests/` |
| 11 | cli (root files) | L3 | LEAF | 4 | 2,942 | Typer CLI (`elspeth run / resume / validate / explain / plugins / purge`) |

Detail per entry — including cited or contradicted KNOW-* claims — is in `02-l1-subsystem-map.md`.

## 3. Open architectural questions (carried forward from discovery)

These are not resolved at L1; they shape L2 dispatch.

1. **Responsibility cut between `contracts/` (L0, 17.4k) and `core/` (L1, 20.8k).** Both are described as "foundation"; ADR-006 explicitly relocated several primitives between the two. The post-ADR-006 boundary should be re-validated in an L2 pass on `contracts/` + `core/`.
2. **`web/composer/` (5,514 LOC across `tools.py`+`state.py`) ↔ `composer_mcp/` (824 LOC).** Whether these are duplicate composer state machines, or whether `composer_mcp/` is a thin transport around `web/composer/`, cannot be determined at L1 depth. The catalog keeps them sibling pending evidence.
3. **Why are `mcp/` and `composer_mcp/` siblings rather than nested?** Catalog entry 6 explicitly argues against merging based on disjoint tool surfaces (`mcp__elspeth-composer__*` vs. analyser tools), separate console scripts, and disjoint runtime concerns. An L2 reviewer should validate the file-count + purpose asymmetry holds.
4. **Plugin count drift in ARCHITECTURE.md.** KNOW-A35 says "25 plugins" while KNOW-A72 (same doc) says "46". Per-category enumeration sums to 25; the 46 figure is unsourced. A doc-correctness pass is needed but is *not* an L2 architecture task.
5. **`engine/orchestrator/core.py` 3,281 LOC + `engine/processor.py` 2,700 LOC + `engine/coalesce_executor.py` 1,603 LOC = ~43% of the engine in three files.** KNOW-A70 already calls this out as a quality risk. Whether the concentration is structural (the orchestrator is doing too much) or accidental (large files but cleanly factored internally) is the central question for the L2 engine pass.

## 4. Doc tensions surfaced (recorded, not resolved)

Five tensions between the institutional documentation set and the verified codebase, all flagged in `02-l1-subsystem-map.md` §Closing:

1. **Plugin-count contradiction** in ARCHITECTURE.md (KNOW-A35 = 25 vs KNOW-A72 = 46).
2. **ADR table staleness** — ARCHITECTURE.md tabulates ADR-001..006 only; ADRs 007..017 are accepted but unindexed.
3. **Schema-mode vocabulary drift** — PLUGIN.md uses `dynamic`/`strict`/`free` in a table (KNOW-P23) but `observed`/`fixed`/`free` in YAML examples (KNOW-P24).
4. **Subsystem-LOC drift** — verified counts diverge ~17% from KNOW-A* figures; ARCHITECTURE.md is roughly one major iteration behind.
5. **`testing/` misidentification (KNOW-A18)** — ARCHITECTURE.md describes `tests/chaos*` (out of scope) under the `testing/` heading; `src/elspeth/testing/` is actually a 2-file pytest-xdist plugin. Recorded as `[DIVERGES FROM KNOW-A18]` in catalog entry 10.

These should be triaged in a **doc-correctness pass**, separate from architectural L2 work.

## 5. Confidence

| Aspect | Confidence | Reason |
|--------|------------|--------|
| Subsystem inventory + sizing | High | `find`-verified counts; layer assignments deterministic |
| Layer model edges | High | CI-enforced, clean run captured |
| Composite/Leaf classification | High | Δ4 heuristic mechanically applied; matches Δ4 expectation |
| Per-subsystem responsibility (8 of 11: `contracts`, `core`, `engine`, `plugins`, `mcp`, `telemetry`, `tui`, `cli`) | High | At least 2 cited KNOW-* claims per subsystem matching observation |
| Per-subsystem responsibility (`web`, `composer_mcp`) | Medium | Institutional docs predate these subsystems; few KNOW-A* claims |
| Per-subsystem responsibility (`testing`) | Medium | Verified, but explicitly contradicts KNOW-A18 |
| L3 ↔ L3 edge structure | **Not assessed at L1** | Deferred (Δ5 grep ban); L2 prerequisite |
| Doc tensions resolution | **Not attempted at L1** | Out of scope; flagged for doc-correctness pass |

## 6. Deferred to later passes (Δ6)

The L1 pass deliberately does **not** cover the following. Each is named with the rationale and the recommended downstream owner.

| Deferred scope | LOC / size | Why deferred at L1 | Recommended owner |
|----------------|-----------:|--------------------|-------------------|
| `src/elspeth/web/frontend/` | ~13k LOC TS/React | Python-lens archaeologist cannot map TSX usefully | A frontend-aware archaeologist (e.g., the `lyra-site-designer` skillpack or a JS/TS-specialised codebase explorer) |
| `tests/` | ~351k LOC, 851 files | Test architecture is a separate deliverable; rolling it into L1 would break the depth cap | A future `05-test-architecture.md` pass — distinct workspace |
| `examples/` | 36 example pipelines | Inventoried by count only; per-pipeline analysis is per-vertical, not architectural | A worked-examples curation pass |
| `scripts/` | ~12k LOC CI/tooling | Only the layer enforcer (Δ5) and the freeze-guard enforcer were touched | A CI/tooling audit pass — only the enforcer scripts are architecturally load-bearing |
| Files >1,500 LOC | 12 files identified | Internals would explode the L1 depth cap | The corresponding L2 cluster pass (per §7 below) |
| L3 ↔ L3 import graph | n/a | Layer-permitted but unconstrained; Δ5 forbids grep | The L2 enumeration cluster (§7, item 6) |
| Doc-correctness reconciliation | 5 tensions in §4 | Documentation correctness is not an architecture task | A separate doc-correctness pass — NOT bundled into any L2 dispatch |

## 7. Recommended L2 dispatch

Six L2 clusters, **prioritised by risk** (deep-dive candidate concentration, structural-question density, downstream blocking effect) — *not* by raw LOC.

### Priority 1 — `engine/` cluster

- **Cluster name:** `l2-engine-core-execution`
- **Scope path:** `src/elspeth/engine/`
- **Estimated effort bracket:** Large (3–5 hours of agent time)
- **Why this priority:** Three deep-dive candidates concentrate ~43% of engine LOC in three files (`orchestrator/core.py` 3,281, `processor.py` 2,700, `coalesce_executor.py` 1,603); KNOW-A70 explicitly flags these as quality risks; this subsystem owns the run lifecycle (token identity, terminal states, cross-check, retry semantics) and ADR-009/010's dispatch-shape work lands here. Any architectural risk in the engine cascades into every plugin.

### Priority 2 — `web/` cluster

- **Cluster name:** `l2-web-composer-and-server`
- **Scope path:** `src/elspeth/web/` (Python only — `frontend/` excluded)
- **Estimated effort bracket:** Large (3–5 hours)
- **Why this priority:** `web/composer/tools.py` (3,804 LOC) is the single largest file in the tree; the sibling-vs-nested question with `composer_mcp/` is open (Q2/Q3 in §3); 8 backend sub-packages are entirely unindexed by ARCHITECTURE.md; the institutional knowledge gap (no KNOW-A* coverage) means this is where unknown-unknowns are most likely.

### Priority 3 — `core/` cluster

- **Cluster name:** `l2-core-foundation-and-landscape`
- **Scope path:** `src/elspeth/core/`
- **Estimated effort bracket:** Large (3–5 hours)
- **Why this priority:** Four deep-dive candidates span `config.py` (2,227), `dag/graph.py` (1,968), and the two largest Landscape repositories (`execution_repository.py` 1,750, `data_flow_repository.py` 1,590); Landscape is the legal record of the system — any architectural risk here is reputational. Six sub-packages need responsibility re-validation post-ADR-006.

### Priority 4 — `plugins/` cluster

- **Cluster name:** `l2-plugins-ecosystem`
- **Scope path:** `src/elspeth/plugins/{infrastructure,sources,transforms,sinks}/` (consider four sub-clusters)
- **Estimated effort bracket:** Large–Very Large (4–8 hours; could reasonably split into four parallel sub-clusters)
- **Why this priority:** Largest subsystem by LOC (30,399) but **already cleanly partitioned** into the four sub-packages; per-category cluster passes are straightforward and parallelisable; only one deep-dive candidate (`azure_batch.py` 1,592). Lower priority than 1–3 because the structural risk-density per LOC is lower (modular by design).

### Priority 5 — `contracts/` cluster

- **Cluster name:** `l2-contracts-leaf-and-types`
- **Scope path:** `src/elspeth/contracts/`
- **Estimated effort bracket:** Medium (2–3 hours)
- **Why this priority:** Only one deep-dive candidate (`errors.py` 1,566); responsibility is well-cited from ADR-006 + multiple KNOW-A* claims; the L0 leaf invariant is mechanically enforced (clean run). The cluster is mainly about boundary re-validation post-ADR-006 and inventorying the freeze-contract surface area.

### Priority 6 — L3 ↔ L3 import-graph enumeration (cross-cutting prerequisite)

- **Cluster name:** `l2-cross-l3-edge-graph`
- **Scope path:** `src/elspeth/{plugins,web,mcp,composer_mcp,telemetry,tui,testing}` + `cli.py / cli_helpers.py / cli_formatters.py` (imports only)
- **Estimated effort bracket:** Small (1–2 hours) — pure import-graph extraction, no semantic analysis
- **Why this priority:** **This is a prerequisite for parts of priority 2 (web ↔ composer_mcp) and 4 (plugins ↔ telemetry).** It is intentionally separated from any single composite cluster because the deliverable is a graph, not a per-subsystem narrative. Recommended tooling: an AST-based import scanner (could extend the existing `enforce_tier_model.py`); explicitly **not** grep, per the discipline of this hierarchical pass.

### Suggested dispatch order

If running passes serially: **6 → 1 → 3 → 2 → 4 → 5**. Cluster 6 unblocks 1/2/3 by providing the L3↔L3 graph. After that, engine/core/web are the high-risk composites; plugins and contracts close out. If parallelising: 6 first solo (it's small), then 1+2+3 in parallel, then 4+5.

## 7.5. Phase 0 amendments to dispatch queue

This section amends §7 with findings from the Phase 0 L3↔L3 import graph oracle (`temp/l3-import-graph.json`, schema v1.0); where §7.5 conflicts with §7, §7.5 is authoritative.

### Findings → amendments

| Finding | Oracle citation | §7 item amended | Amendment |
|---------|-----------------|-----------------|-----------|
| F1. `composer_mcp/` is coupled to `web/composer/`, not an independent sibling. | `[ORACLE: edge composer_mcp → web/composer, weight 12, type_checking_only=false, conditional=false; sample sites composer_mcp/server.py:28, :29]` | §7 Priority 2 (`web/` cluster) | Cluster renamed to **composer cluster (web/ + composer_mcp/)**; scope path expanded; effort raised. |
| F2. `mcp/` and `composer_mcp/` are independent siblings (catalog entry 6's "do not merge" guidance was correct). | `[ORACLE: edges with from='mcp' and to startswith 'composer_mcp' = 0; edges with from startswith 'composer_mcp' and to='mcp' = 0]` | §7 — no item amended; closes L1 open question Q3. | Recorded under "Closed L1 open questions" below. |
| F3. `plugins/infrastructure/` is the spine of the plugins ecosystem; sinks/sources/transforms are clients of it. | `[ORACLE: edge plugins/sinks → plugins/infrastructure, weight 45 (heaviest single L3 edge); edge plugins/transforms → plugins/infrastructure, weight 40]` | §7 Priority 4 (`plugins/`) | Adds a **Reading order** subfield: `plugins/infrastructure/` FIRST, then sinks/sources/transforms as clients. Effort bracket UNCHANGED. |
| F4. The web/* sub-packages form a 7-node strongly-connected component; no acyclic decomposition is possible. | `[ORACLE: stats.scc_count = 5, stats.largest_scc_size = 7; strongly_connected_components[4] = ['web', 'web/auth', 'web/blobs', 'web/composer', 'web/execution', 'web/secrets', 'web/sessions']]` | §7 Priority 2 (now "composer cluster") | Effort bracket revised UP from "Large (3–5 hr)" to "Very Large (5–7 hr)" — the SCC must be analysed as a unit, not as 7 independent sub-areas. |
| F5. ~97% of L3 edges are unconditional runtime coupling (0 TYPE_CHECKING-only, 2 conditional out of 77). | `[ORACLE: stats.total_edges = 77, stats.type_checking_edges = 0, stats.conditional_edges = 2, stats.reexport_edges = 0]` | All §7 items (cross-cutting). | Recorded under "Standing note for all L2 passes" below. |

### Revised L2 dispatch queue

Risk-weighted ordering from §7 is preserved. Specific item-level changes:

#### Priority 1 — `engine/` cluster
Unchanged from §7 Priority 1.

#### Priority 2 — composer cluster (web/ + composer_mcp/)  *(RENAMED from "web/ cluster")*

- **Cluster name:** `l2-composer-cluster-web-and-composer-mcp`
- **Scope path:** `src/elspeth/web/` + `src/elspeth/composer_mcp/` (Python only — `web/frontend/` excluded)
- **Estimated effort bracket:** **Very Large (5–7 hr)** — revised UP from §7's "Large (3–5 hr)". Justification: F4's 7-node SCC means the web sub-packages cannot be analysed independently in 7 parallel slices; they must be reasoned about as one tangled cluster, raising both reading and synthesis cost.
- **Why this priority:** F1 demonstrates `composer_mcp/` substantively imports from `web/composer/` (`[ORACLE: edge composer_mcp → web/composer, weight 12]`); the L1 catalog's sibling/sibling presumption for the two MCP surfaces was correct for `mcp/` vs `composer_mcp/` (F2) but wrong for `composer_mcp/` vs `web/composer/` — the latter pair shares the composer state machine and must be scoped together. The 7-node web/* SCC (F4) compounds this: the composer cluster cannot be decomposed acyclically.

#### Priority 3 — `core/` cluster
Unchanged from §7 Priority 3.

#### Priority 4 — `plugins/` cluster

- All §7 Priority 4 fields preserved, plus:
- **Reading order:** `plugins/infrastructure/` FIRST, then `plugins/sinks/`, `plugins/sources/`, `plugins/transforms/` as clients of it. Justification: `[ORACLE: edge plugins/sinks → plugins/infrastructure, weight 45]` (heaviest single L3 edge in the entire codebase) and `[ORACLE: edge plugins/transforms → plugins/infrastructure, weight 40]` confirm the infrastructure sub-package is the plugin ecosystem's spine; reading sources-first or transforms-first means re-reading infrastructure once it's referenced. Effort bracket UNCHANGED.

#### Priority 5 — `contracts/` cluster
Unchanged from §7 Priority 5.

### Completed prerequisites

- **§7 Priority 6 — L3↔L3 import-graph enumeration** is **COMPLETED** (Phase 0). Artefacts available at:
  - `temp/l3-import-graph.json` (schema v1.0, deterministic with `--no-timestamp`)
  - `temp/l3-import-graph.mmd` (Mermaid `flowchart LR`)
  - `temp/l3-import-graph.dot` (Graphviz, renders via `dot -Tsvg`)
  - The dispatch queue is no longer 6-deep; 5 cluster items remain.

### Closed L1 open questions

- **Q2 (web/composer ↔ composer_mcp):** coupled, see F1. The two share the composer state machine; a single L2 pass must scope them together.
- **Q3 (mcp/ vs composer_mcp/ independence):** confirmed independent, see F2. They share MCP transport but no import edges; catalog entries 6/7's "do not merge" guidance is upheld.

### Still open after Phase 0

These §3 questions were not resolved by the L3↔L3 import graph (they require code-reading, not graph topology):

- **Q1** — Responsibility cut between `contracts/` (L0) and `core/` (L1) post-ADR-006.
- **Q4** — Plugin-count drift in ARCHITECTURE.md (KNOW-A35 = 25 vs KNOW-A72 = 46). Not a graph question; requires a doc-correctness pass.
- **Q5** — Whether `engine/orchestrator/core.py` (3,281 LOC) + `engine/processor.py` (2,700 LOC) + `engine/coalesce_executor.py` (1,603 LOC) concentration is structural or accidental. The L2 engine pass (Priority 1) owns this.

### Standing note for all L2 passes

L2 archaeologists do **NOT** need to invest effort hunting for TYPE_CHECKING-guarded or conditionally-imported coupling at the L3 boundary; the oracle confirms ~97% of L3 edges are unconditional runtime coupling (`[ORACLE: stats.total_edges = 77, stats.type_checking_edges = 0, stats.conditional_edges = 2]`). Hidden coupling, if any, lives below the subsystem boundary and is in-scope for per-cluster L2 work, not for cross-cluster suspicion.

### Suggested launch order (revised)

§7's original suggested order was `6 → 1 → 3 → 2 → 4 → 5`, with item 6 as the prerequisite unblocker. With item 6 now complete and the composer-cluster re-scope absorbed:

- **Serial:** 1 (engine/) → 3 (core/) → 2 (composer cluster) → 4 (plugins/) → 5 (contracts/).
  Rationale: items 1 and 3 are oracle-independent and the highest-risk-density clusters, so they go first. Item 2 (composer cluster) is now larger and benefits from launching after 1+3 so any cross-references into engine/core are already mapped.
- **Parallel:** 1+3 first (oracle-independent), then 2+4, then 5.
  Rationale: 1 and 3 share no cluster boundary with each other or with 2/4/5 at the L3 edge level, so they parallelise cleanly. 2+4 both depend on understanding the L2 engine surface and so should follow 1+3.

### Observations deferred to L2

While verifying F1–F5 in the JSON, these additional oracle observations were noted but are **NOT amendments** — they are inputs for L2 cluster passes:

- `[ORACLE: edge mcp/analyzers → mcp, weight 29]` — third-heaviest single edge; the MCP analyser sub-package has an inverted dependency on the parent `mcp/` namespace, contributing to SCC#1. The L2 mcp pass (if undertaken) should treat `mcp/` and `mcp/analyzers/` as a 2-node cluster, mirroring the composer-cluster pattern at smaller scale.
- `[ORACLE: stats.reexport_edges = 0]` — zero edges aggregated as re-exports under the AND-rule. The package-vs-subpackage SCCs (#1, #2, #3, #4) are real runtime cycles, not just re-export artefacts. The L2 cluster passes for `mcp/`, `plugins/transforms/llm/`, `telemetry/`, and `tui/` should triage whether these cycles are intentional re-export conventions or unintentional structural coupling.
- `[ORACLE: '.' top-level node present]` — the cli root files (`cli.py`, `cli_helpers.py`, `cli_formatters.py`, `__init__.py`) collapse to the `.` node in the graph (file's parent is the root). Edges from `.` (e.g., `[ORACLE: edge . → plugins/infrastructure, weight 7]`) confirm the cli root has direct dependencies on `plugins/infrastructure/` — consistent with KNOW-P22's `TRANSFORM_PLUGINS` registry living in `cli.py`.

## 8. Limitations of this L1 pass (for the record)

Documented honestly because the cite-or-contradict discipline depends on this section being read by future passes:

- **Depth cap held.** No file >1,500 LOC was opened. No per-file enumeration in any subsystem entry. Twelve such files are flagged for L2 candidates.
- **Grep ban held.** No L3↔L3 imports were derived by grep. The clean enforcer status is the only dependency-truth signal used; everything else is layer-derived.
- **Time budget held.** The pass completed within the Δ8 90-minute target. No 2-hour tripwire fired.
- **Two divergences from the scope override** — both surfaced explicitly:
  1. Subsystem count is 11 not 12 (Δ4 expected-classification list confirms 11; treated as prose typo).
  2. The Δ5-named tool `enforce_tier_model.py` is *both* the trust-tier defensive-pattern scanner (CLAUDE.md sense) *and* the layer-import enforcer (ADR-006 Phase 5 sense); its clean output is sufficient for cross-layer dependency truth but does not enumerate the dependency graph. Used the path→layer schema in the script (lines 237–248: `LAYER_HIERARCHY` 237–241 + `LAYER_NAMES` 243–248) as the authoritative cross-layer source. Recorded in `00-coordination.md`.
- **Validator's contract** (Δ7) reduced for this pass: only inventory parity, dependency-claim consistency with the enforcer, composite/leaf heuristic application, citation resolution, and deferral presence. Per-file claims and component-level diagram details are out of scope.
