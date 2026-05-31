# 00 — Coordination Plan

## Pass identity

**Pass:** L1 shallow map (Option B + Option E "Very-Large Hierarchical, L1 leg")
**Workspace:** `docs/arch-analysis-2026-04-29-1500/`
**Date opened:** 2026-04-29
**Time budget (Delta 8):** 60–90 minutes target, 2-hour hard tripwire.

The deliverable menu was pre-decided per scope override (Delta 1). No `AskUserQuestion` was issued.

## Scope

- **In:** `src/elspeth/` top-level subsystems (Python production code, ~121k LOC).
- **Out (deferred per Delta 6):**
  - `src/elspeth/web/frontend/` (~13k LOC TS/React) → frontend-aware archaeologist.
  - `tests/` (~351k LOC) → separate test-architecture pass.
  - `examples/` (36 pipelines) → noted by count, not analysed individually.
  - `scripts/` (~12k LOC) → only the layer/freeze enforcers are touched (Delta 5).
  - Any single file >1,500 LOC → flagged as L2-deep-dive candidate, not summarised.

## Subsystem inventory (verified against `src/elspeth/`)

Counts captured 2026-04-29. Per-directory file counts and LOC ran cleanly with `find ... -print0 | xargs -0 cat | wc -l`.

| # | Subsystem | Files (.py) | LOC | Layer (per oracle) | Composite/Leaf (Δ4 heuristic) |
|---|-----------|------------:|----:|--------------------|------------------------------:|
| 1 | `contracts/` | 63 | 17,403 | L0 | **COMPOSITE** |
| 2 | `core/` | 49 | 20,791 | L1 | **COMPOSITE** |
| 3 | `engine/` | 36 | 17,425 | L2 | **COMPOSITE** |
| 4 | `plugins/` | 98 | 30,399 | L3 | **COMPOSITE** |
| 5 | `web/` | 72 | 22,558 | L3 | **COMPOSITE** |
| 6 | `mcp/` | 9 | 4,114 | L3 | LEAF |
| 7 | `composer_mcp/` | 3 | 824 | L3 | LEAF |
| 8 | `telemetry/` | 14 | 2,884 | L3 | LEAF |
| 9 | `tui/` | 9 | 1,175 | L3 | LEAF |
| 10 | `testing/` | 2 | 877 | L3 | LEAF |
| 11 | `cli (root files)` | 4 | 2,942 | L3 | LEAF (with cli.py L2-deep-dive flag at 2,357 LOC) |

**Total verified: 121,392 LOC across 11 subsystems** (matches user's "~121k LOC").
The scope-override prose said "12 top-level subsystems" but the explicit composite/leaf list in Delta 4 enumerates only 11; treating the prose count as a typo and proceeding with 11. This is recorded as `[DIVERGES FROM]` in the L1 map.

Composite/leaf classification heuristic (Delta 4): COMPOSITE iff (≥4 sub-packages OR ≥10k LOC OR ≥20 files). Verified mechanically; matches the expected list in Delta 4 exactly (5 COMPOSITE, 6 LEAF).

## Strategy

**Sequential with delegated parallelism.** Coordinator (this conversation) does not read source files; subagents do. Wave structure:

| Wave | Tasks | Parallelism |
|------|-------|-------------|
| W1 | Knowledge ingestion → `00b-existing-knowledge-map.md`; Discovery scan → `01-discovery-findings.md` | parallel (independent inputs) |
| W2 | L1 catalog → `02-l1-subsystem-map.md` | sequential after W1 |
| W3 | C4 context diagram → `03-l1-context-diagram.md`; Summary → `04-l1-summary.md` | diagram first, then summary |
| W4 | Validation → `temp/validation-l1.md` | sequential after W3 |

## Dependency oracle (Delta 5)

**Tool:** `scripts/cicd/enforce_tier_model.py` ran clean (`Check passed`). Raw + layer schema + interpretation captured in `temp/tier-model-oracle.txt`.

**Important discrepancy surfaced:** Delta 5 calls this script "the dependency truth-source". The script implements *both* trust-tier defensive-pattern checks AND layer-import checks (rule_id `L1` for runtime upward, `TC` for TYPE_CHECKING). On a clean codebase the script emits no findings, so its output alone does not enumerate the dependency graph. The authoritative graph is derived from:
1. The path→layer table inside the script (lines 237–247).
2. The clean check result, which proves no layer violations exist.

This gives subsystem-level dependency truth at *cross-layer* granularity. **L3↔L3 edges (e.g., `web` ↔ `plugins`, `mcp` ↔ `core`) are layer-permitted but unconstrained**; the L1 catalog will mark them deferred to L2 cluster passes rather than hand-derive them with grep (which Delta 5 forbids).

## Execution log

- 2026-04-29 15:00 — Workspace created at `docs/arch-analysis-2026-04-29-1500/`.
- 2026-04-29 15:01 — Ran `enforce_tier_model.py` per Delta 5. Exit 0, "Check passed". Oracle artefact captured with layer schema and interpretation at `temp/tier-model-oracle.txt`.
- 2026-04-29 15:02 — Verified subsystem inventory: 11 subsystems totalling 121,392 LOC. Composite/leaf classification confirmed mechanically.
- 2026-04-29 15:03 — Surfaced enforcer scope discrepancy (Δ5 expected dependency-graph dump; tool produces violation-list). Decision: use layer schema + clean status as cross-layer truth; defer L3↔L3 edges to L2 passes. Recorded in this plan.
- 2026-04-29 15:04 — Coordination plan written. Wave 1 dispatch prepared.
- 2026-04-29 15:10 — Wave 1 closed. `00b-existing-knowledge-map.md` (~250 atomic claims, 293 lines, no Superseded ADRs but ADR-007/008 amended by 009/010, ADR-009 Clause 3 retired by 012, ADR-015 retired). `01-discovery-findings.md` written (LOC bimodal: 5 composites = ~89%; 12 files >1,500 LOC; entry points confirmed verbatim from pyproject `[project.scripts]`).
- 2026-04-29 15:25 — Wave 2 closed. `02-l1-subsystem-map.md` written (192 lines, 11 entries, 14 [CITES] + 6 [DIVERGES FROM] markers, 12 deep-dive files flagged, 0 Low / 2 Medium confidence: `web/` and `composer_mcp/` due to absence of KNOW-A* coverage).
- 2026-04-29 15:30 — Wave 3 closed. `03-l1-context-diagram.md` (System Context view + Container view by layer, 27 layer-enforced edges, L3↔L3 explicitly deferred). `04-l1-summary.md` (with 6-cluster L2 dispatch queue, prioritised by risk-density: engine > web > core > plugins > contracts, plus a cross-cutting L3↔L3 import-graph cluster as prerequisite).
- 2026-04-29 15:40 — Wave 4 closed. Validation `temp/validation-l1.md`: **APPROVED**. 0 CRITICAL, 0 WARNING, 1 MINOR (cosmetic line-number drift). MINOR addressed: catalog/diagram/summary now uniformly cite `enforce_tier_model.py:237–248`. L1 pass complete; ready for L2 dispatch.
- [2026-04-29T10:00:00Z] Phase 9 doc-correctness pass landed. Resolved L1-deferred tensions T1–T5 (plugin-count drift, ADR-table staleness, schema-mode vocabulary drift, ARCHITECTURE.md LOC drift, KNOW-A18 testing/↔tests/ conflation). Edits made to ARCHITECTURE.md, CLAUDE.md, PLUGIN.md (AGENTS.md verified clean — no T5 conflation site present). Ground truth recorded in `temp/doc-correctness-ground-truth.md`. 13 follow-up observations deferred in `temp/doc-correctness-deferrals.md`. Validator: APPROVED.

## Phase 0 — L3↔L3 import-graph oracle (completed 2026-04-29 02:32)

The L1 dispatch queue's prerequisite item 6 ("L3↔L3 import-graph enumeration") is now landed as an additive `dump-edges` subcommand on `scripts/cicd/enforce_tier_model.py`. The existing `check` subcommand is regression-free (re-verified clean) and the new subcommand always exits 0 (graph content is observational, not enforcement).

**Modified files:**
- `scripts/cicd/enforce_tier_model.py` — added `dump-edges` subcommand (~440 LOC additive in two new sections: scanner+formatters and CLI dispatch). Existing `check` logic untouched.
- `tests/unit/scripts/cicd/test_enforce_tier_model_dump_edges.py` (NEW) — 12 tests (10 Δ7-mandated cases + 2 formatter smoke tests). All 12 pass.

**Artefacts (in `temp/`, deterministic JSON via `--no-timestamp`-able schema):**
- `temp/l3-import-graph.json` — 38 KB, schema_version 1.0, tool_version sha256:c9cef30d1f6a
- `temp/l3-import-graph.mmd` — 5.7 KB, Mermaid `flowchart LR` with subsystem subgraphs
- `temp/l3-import-graph.dot` — 7.2 KB, Graphviz `digraph` (renders via `dot -Tsvg`)

**Stats from the live tree (`src/elspeth`, L3-only, collapse-to-subsystem ON):**

| Metric | Value |
|--------|------:|
| total_nodes | 33 |
| total_edges | 77 |
| type_checking_edges | 0 |
| conditional_edges | 2 |
| reexport_edges | 0 |
| scc_count | **5** |
| largest_scc_size | **7** |

**Strongly-connected components** (the headline finding — direct empirical evidence of cluster-boundary leakage that the L2 dispatch wave should triage):

| # | Size | Members | Interpretation |
|--:|----:|---------|----------------|
| 1 | 2 | `mcp` ↔ `mcp/analyzers` | Package-vs-subpackage cycle (typical re-export pattern) |
| 2 | 2 | `plugins/transforms/llm` ↔ `plugins/transforms/llm/providers` | Same — LLM transforms ↔ providers register through the parent package |
| 3 | 2 | `telemetry` ↔ `telemetry/exporters` | Same — exporters import the telemetry namespace |
| 4 | 3 | `tui` ↔ `tui/screens` ↔ `tui/widgets` | Three-way TUI shell coupling |
| 5 | **7** | `web` ↔ `web/auth` ↔ `web/blobs` ↔ `web/composer` ↔ `web/execution` ↔ `web/secrets` ↔ `web/sessions` | **The largest tangle.** Every backend `web/*` sub-package is mutually coupled. The L1 map's Q2 ("`web/composer/` ↔ `composer_mcp/` boundary") and the L2 web cluster pass land directly here. |

**Other notable edges** observed during smoke (not exhaustive — see JSON):
- `composer_mcp → web/composer` weight 12 — directly answers L1 Q3 (`mcp/` vs `composer_mcp/` are NOT a clean sibling/sibling split: `composer_mcp` substantively imports from `web/composer`, so any L2 web pass must scope the composer state machine across both surfaces).
- `plugins/sinks → plugins/infrastructure` weight 45 (heaviest single edge) — confirms the infrastructure sub-package is the plugin ecosystem's spine.
- `plugins/transforms → plugins/infrastructure` weight 40 (second heaviest).
- `mcp/analyzers → mcp` weight 29 — the analysers form an inverted dependency on the parent package, contributing to SCC#1.

**Δ9 gate compliance:**
- `mypy scripts/cicd/enforce_tier_model.py` — clean (no errors)
- `ruff check scripts/cicd/enforce_tier_model.py tests/unit/scripts/cicd/test_enforce_tier_model_dump_edges.py` — clean
- `enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` — exit 0, "Check passed" (regression-free)

**Δ10 budget:** Implementation + tests + artefacts produced within budget; no scope drift into refactoring or graph analysis (latter belongs to L2).

**L2 dispatch update (informational, not modifying `04-l1-summary.md` per Δ8):**
The L2 wave can now consume the JSON directly (or run `dump-edges --no-timestamp` to re-derive deterministically). SCC#5's 7-node web/* tangle is the highest-leverage L2 finding — recommend the L2 web cluster pass treat `composer_mcp/` and `web/composer/` as **a single composer cluster** rather than maintaining the L1 sibling/sibling presumption. The non-web SCCs (1–4) are all `package ↔ subpackage` re-export patterns and are likely benign (Python convention) but should be confirmed by the L2 cluster passes for those subsystems.

- 2026-04-29T02:45:00Z — Phase 0.5 amendment landed. Added §7.5 to 04-l1-summary.md reflecting Phase 0 oracle findings F1–F5. §7 unchanged; oracle artefacts unchanged; L1 catalog/diagram untouched. Closed L1 open questions: Q2 (composer coupling), Q3 (mcp independence). Revised L2 #2 scope and budget; revised L2 #4 reading order; marked L2 #6 complete. Standing note added for all L2 passes (F5: unconditional runtime coupling).
- 2026-04-29T07:30:00Z — Phase 8 stitching pass landed. Synthesised five L2 cluster reports (engine, core, composer, plugins, contracts) into 99-stitched-report.md (~6,933 words, 48 claims, 119 citations) and 99-cross-cluster-graph.md (Container + Component Mermaid diagrams, ≤1,500 words). Reconciliation log: 0 cross-cluster contradictions, 2 reconciled near-misses (R1 F1 framing depth, R2 TYPE_CHECKING accounting methodology), 4 already-resolved [DIVERGES FROM] divergences flowed to doc-correctness pack, 14 already-aligned cross-cluster handshakes confirmed, 1 propagation hazard (contracts cluster KNOW-A* citation editorial defect) recorded. Validator returned APPROVED (V1–V9 all PASS) with 2 deferred WARNINGs (V5: 3 §3 sub-claims with single-source citations honestly disclosed Medium confidence in §10 ledger; V2: S4.6 attribution mis-labels intra-cluster JSON as Phase 0 oracle) and 1 MINOR finding (2 manifest entries curated out of stitched report — permitted under synthesis discipline). Warnings deferred to downstream pack consumers per APPROVED verdict, not blocking. Frozen-input check: PASS (only the 5 expected NEW files added under workspace; no L1 deliverable, cluster workspace, or Phase 0 artefact mutated). Recommended downstream packs: axiom-system-architect (architecture critique + improvement roadmap), ordis-security-architect (threat modelling on trust-tier topology + audit-trail completeness), axiom-system-archaeologist L3 deep-dives (6+ files >1,500 LOC: tools.py, state.py, processor.py, config.py, dag/graph.py, errors.py, orchestrator/core.py), doc-correctness pass (KNOW-A24 / KNOW-A35-vs-A72 / contracts citation editorial defect), cross-cluster integration-tier audit (KNOW-C44 production-path conformance).
