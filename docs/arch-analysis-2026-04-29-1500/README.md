# ELSPETH Architecture Analysis — 2026-04-29

A **hierarchical, evidence-anchored architecture analysis** of the ELSPETH
codebase produced by adapting the `axiom-system-archaeologist` skillpack
for a very-large repository (~121k production Python LOC, 11 top-level
subsystems). The workspace is a **frozen audit artefact set** — deliverables
here are produced once and consumed unchanged.

The set covers four contracted deliverables:

1. **Discovery & inventory** (the codebase shape and what's in it)
2. **Hierarchical analysis** (5 cluster reports + 1 synthesis report)
3. **Deterministic L3 import oracle** (a reproducible coupling artefact)
4. **Architecture quality assessment** (severity-rated findings + roadmap)

This README is the **navigation hub**. If you read nothing else, read
the matrix in §2 and pick a starting document.

---

## §1 What this analysis does and does not cover

### Covered

- All ~121,408 LOC of production Python under `src/elspeth/`
- The 4-layer model and its CI enforcement (`enforce_tier_model.py`)
- All 11 top-level subsystems, classified COMPOSITE / LEAF
- The L3 import topology (33 nodes, 77 edges, 5 SCCs) as a deterministic
  oracle (`temp/l3-import-graph.json`)
- 5 cluster-level deep dives (engine, core, composer, plugins, contracts)
- Cross-cluster reconciliation (14 handshakes, 0 contradictions)
- Severity-rated quality assessment with prioritised improvement roadmap

### Explicitly NOT covered (named limitations)

| Out-of-scope area | Size | Why deferred | Owner |
|---|---:|---|---|
| `src/elspeth/web/frontend/` | ~13k LOC TS/React | Python-lens archaeologist cannot map TSX usefully | Frontend-aware archaeologist (`lyra-site-designer` or equivalent) |
| `tests/` | ~351k LOC, ~851 files | Test architecture is a separate deliverable | Test-architecture pass (`ordis-quality-engineering:analyze-pyramid` or equivalent) |
| `examples/` | 36 pipelines | Per-vertical, not architectural | Worked-examples curation pass |
| `scripts/` (beyond enforcers) | ~12k LOC | Only architecturally-relevant scripts touched | CI/tooling audit pass |
| Per-file cohesion of 13 ≥1,500-LOC files | n/a | L3 deep-dive depth | Per-file `axiom-system-archaeologist` deep-dives |

**These gaps are named, not hidden.** Any "complete architecture" claim
about ELSPETH requires the frontend pass and the test-architecture pass
to land first. See `05-quality-assessment.md` §4.5 for the consequences
of the frontend gap.

---

## §2 Document matrix — pick your entry point

| If you want… | Read | Approx. read time |
|---|---|---|
| **The verdict** — quality assessment with severity-rated findings + 12-recommendation roadmap | [`05-quality-assessment.md`](05-quality-assessment.md) | 25 min |
| **The synthesis** — 6,933 words, 119 citations, 48 cross-cluster claims | [`99-stitched-report.md`](99-stitched-report.md) | 30 min |
| **The picture** — Container + Component diagrams synthesising the L3 topology | [`99-cross-cluster-graph.md`](99-cross-cluster-graph.md) | 5 min |
| **The L1 inventory** — 11 subsystems at a glance | [`02-l1-subsystem-map.md`](02-l1-subsystem-map.md) | 15 min |
| **The L1 dispatch trail** — original prioritised cluster queue + Phase 0 amendments | [`04-l1-summary.md`](04-l1-summary.md) | 12 min |
| **The L1 holistic scan** — codebase shape, top-10 large files, layer flows | [`01-discovery-findings.md`](01-discovery-findings.md) | 8 min |
| **A specific cluster** | `clusters/<name>/04-cluster-report.md` for `engine`, `core`, `composer`, `plugins`, `contracts` | 10–15 min each |
| **The L3 import oracle** — machine-readable, schema v1.0 | [`temp/l3-import-graph.json`](temp/l3-import-graph.json) (or `.mmd` / `.dot` for visualisation) | n/a |
| **Indexed institutional knowledge** — ~250 KNOW-* atomic claims from prior docs | [`00b-existing-knowledge-map.md`](00b-existing-knowledge-map.md) | reference |
| **Process & methodology** — coordination plan + execution log across all phases | [`00-coordination.md`](00-coordination.md) | 10 min |

### Recommended reading order

- **For decisions** (architecture pack, security pack, planner):
  `05-quality-assessment.md` → drill into specific findings via cluster
  reports → consult `temp/l3-import-graph.json` for coupling questions.
- **For deep understanding** (new technical lead):
  `01-discovery-findings.md` → `02-l1-subsystem-map.md` →
  `99-stitched-report.md` → `05-quality-assessment.md`.
- **For specific subsystem work**:
  `clusters/<name>/04-cluster-report.md` → that cluster's
  `02-cluster-catalog.md` → relevant code.

---

## §3 Workspace layout

```text
docs/arch-analysis-2026-04-29-1500/
├── README.md                         (this file — navigation hub)
├── 00-coordination.md                Coordination plan + execution log (all phases)
├── 00b-existing-knowledge-map.md     ~250 KNOW-* claims indexed from prior docs
├── 01-discovery-findings.md          L1 holistic scan (codebase shape, large files)
├── 02-l1-subsystem-map.md            11 top-level subsystem catalog entries
├── 03-l1-context-diagram.md          C4 System Context + Container view
├── 04-l1-summary.md                  L1 dispatch queue + Phase 0 amendments §7.5
├── 05-quality-assessment.md          ★ Quality assessment + improvement roadmap
├── 99-stitched-report.md             ★ Synthesis report (the analytical deliverable)
├── 99-cross-cluster-graph.md         System-level architecture diagrams
├── clusters/                         Per-cluster L2 deep-dive workspaces
│   ├── engine/                       L2 #1 — orchestrator + executors + processor
│   ├── core/                         L2 #3 — landscape, dag, config, canonical, etc.
│   ├── composer/                     L2 #2 — web/ + composer_mcp/ (7-node SCC)
│   ├── plugins/                      L2 #4 — infrastructure, sources, transforms, sinks
│   └── contracts/                    L2 #5 — protocols, types, schema contracts
└── temp/                             Oracles, manifests, validators (see §6 semantics)
```

Each `clusters/<name>/` follows an identical internal layout:

```text
clusters/<name>/
├── 00-cluster-coordination.md        Coordination for the cluster pass
├── 01-cluster-discovery.md           Holistic scan of the cluster
├── 02-cluster-catalog.md             Sub-subsystem catalog (one level deeper than L1)
├── 03-cluster-diagrams.md            Cluster-scoped C4 Container + Component
├── 04-cluster-report.md              Cluster-level synthesis (top-3 confidence + uncertainty)
└── temp/                             Cluster-scoped oracle subsets and validators
```

---

## §4 Headline findings

These are the load-bearing claims the workspace produced. Full evidence
chains live in `99-stitched-report.md` §3–§7 and `05-quality-assessment.md` §3.

1. **The 4-layer model is mechanically clean today.** Zero L1
   upward-import violations, zero TYPE_CHECKING layer warnings, across
   every cluster scope. CI-enforced by `enforce_tier_model.py`.
   *(Live re-validated 2026-04-29: still clean.)*

2. **A 7-node strongly-connected component spans every `web/*`
   sub-package.** No acyclic decomposition is possible within `web/`
   in its current shape. Decomposition is non-trivial (the SCC is the
   FastAPI app-factory pattern made structural). *Source:
   `temp/l3-import-graph.json` `strongly_connected_components[4]`.*

3. **`plugins/infrastructure/` is the structural spine of the plugin
   ecosystem.** The `plugins/sinks → plugins/infrastructure` edge has
   weight 45 — the heaviest single L3 edge in the codebase.
   Sinks/sources/transforms are clients, not peers. *Source: same JSON,
   `.edges`.*

4. **`composer_mcp/` is not a sibling of `mcp/`.** It imports
   `web/composer/` at weight 12; the L1 sibling/sibling framing was
   wrong and the L2 cluster pass corrected it.

5. **75 of 77 L3 edges are unconditional runtime coupling.** Zero
   `TYPE_CHECKING`-only edges; only 2 conditional. The codebase does not
   hide its dependencies. Lazy-import patterns are essentially absent at
   the L3 boundary.

6. **The terminal-state-per-token audit invariant is structurally
   guaranteed**, not conventional. Implemented via
   `engine/executors/state_guard.py:NodeStateGuard` (a context-manager
   pattern, locked by tests). *Source:* `clusters/engine/04-cluster-report.md`
   §"Highest-confidence claims" item 3.

7. **The cross-cluster join is structurally sound.** Five
   independently-produced cluster reports agreed at every named
   cross-cluster boundary; 0 contradictions surfaced during reconciliation.

---

## §5 Findings the assessment surfaced (beyond the synthesis)

Live re-validation during quality assessment (`05-quality-assessment.md` §5)
surfaced additions to the synthesis's catalogued findings. Recorded here
so they are visible from the navigation hub:

- **13 files ≥1,500 LOC**, not the 7 named in synthesis §8. Five
  additional files (`cli.py` 2357, `execution_repository.py` 1750,
  `azure_batch.py` 1592, `data_flow_repository.py` 1590,
  `coalesce_executor.py` 1603) are flagged at L1 or resolved at L2 but
  not pulled forward into the synthesis's open-questions section.
- **`web/sessions/routes.py` (1,563 LOC) was missed at L1 entirely.**
  L1 deferral list claimed "12 files identified" but the live tree has
  13. Inventory completeness defect.
- **The synthesis-flagged ADR-010 dispatcher verification gap (synthesis
  §5.2 / §7.2) was closed during quality assessment.** Direct L3 read
  of `src/elspeth/engine/executors/declaration_dispatch.py:120–172`
  confirms both except branches correctly append to the violations
  list (audit-complete-with-aggregation per the docstring); 1,923 LOC of
  dedicated test coverage exists across unit / property / integration.
  See `05-quality-assessment.md` §3.1 E1 for the verification record.

---

## §6 `temp/` semantics

The `temp/` directories contain two distinct file classes — **do not
delete or relocate without distinguishing them**:

| Class | Examples | Durability | Cited by other deliverables? |
|---|---|---|---|
| **Oracle artefacts** | `temp/l3-import-graph.{json,mmd,dot}`, `clusters/*/temp/intra-cluster-edges.json`, `clusters/*/temp/layer-check-*.txt`, `temp/tier-model-oracle.txt` | Frozen, durable | Yes — cited by JSON path or filename throughout |
| **Pass workings** | `temp/synthesis-input-manifest.md`, `temp/reconciliation-log.md`, `temp/validation-*.md`, `temp/doc-correctness-*.md` | Single-use | No — consumed once at validation time |

Both classes are kept under `temp/` for workspace tidiness; the durability
distinction is documented here rather than enforced by directory
structure.

---

## §7 Phase chronology

The workspace was produced over a sequence of disciplined phases, each
with its own scope-override prompt, time budget, validator gate, and
citation contract. Phase boundaries are recorded in the
[execution log](00-coordination.md#execution-log).

| Phase | Date | Output |
|---|---|---|
| **L1 shallow map** | 2026-04-29 (~40 min) | 6 deliverables; 11 subsystems classified COMPOSITE vs LEAF |
| **Phase 0 — L3 oracle** | 2026-04-29 02:32Z | `enforce_tier_model.py dump-edges` extension + 12 tests + 3 graph artefacts |
| **Phase 0.5 — L1 amendment** | 2026-04-29 02:45Z | `04-l1-summary.md` §7.5 added (oracle findings F1–F5) |
| **L2 #1 — engine** | 2026-04-29 | `clusters/engine/` |
| **L2 #2 — composer** | 2026-04-29 | `clusters/composer/` (7-node SCC analysed as unit) |
| **L2 #3 — core** | 2026-04-29 | `clusters/core/` |
| **L2 #4 — plugins** | 2026-04-29 | `clusters/plugins/` (infrastructure-first reading order) |
| **L2 #5 — contracts** | 2026-04-29 | `clusters/contracts/` |
| **Phase 8 — stitching** | 2026-04-29 07:30Z | `99-stitched-report.md` + `99-cross-cluster-graph.md` |
| **Phase 9 — doc-correctness** | 2026-04-29 | T1–T5 resolved across `ARCHITECTURE.md`, `CLAUDE.md`, `PLUGIN.md`; 13 deferrals |
| **Phase 10 — quality assessment** | 2026-04-29 | `05-quality-assessment.md` (this pass): severity-rated findings + 12 recommendations + scope-gap honesty |

---

## §8 Re-running the L3 oracle

The graph is deterministic. To re-derive after codebase changes:

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py dump-edges \
  --root src/elspeth \
  --format json \
  --output /tmp/l3-import-graph-fresh.json \
  --no-timestamp
```

`--no-timestamp` produces byte-identical output across runs given the
same source tree, so `diff` against the workspace artefact will show only
true import-graph drift.

**Re-validation status as of `05-quality-assessment.md` (2026-04-29):**
the fresh dump is byte-identical to `temp/l3-import-graph.json` except
for the `generated_at` and `tool_version` metadata fields, confirming
the synthesis's structural claims still hold.

---

## §9 Recommended downstream packs

In priority order. Mapped to specific findings in `05-quality-assessment.md` §6:

1. **`ordis-security-architect`** — STRIDE threat model on the
   trust-tier topology + audit-trail completeness verification. *(R8.)*
2. **Frontend archaeologist** — `lyra-site-designer` or TS/React
   specialist; covers the ~13k-LOC SPA gap. *(R6.)*
3. **`ordis-quality-engineering:analyze-pyramid`** — test-architecture
   pass; resolves the inverted-pyramid risk on the 2.9× src/-to-tests
   ratio. *(R7.)*
4. **`axiom-system-archaeologist` per-file deep-dives** — covers the
   13 ≥1,500-LOC files identified during assessment, prioritising
   `processor.py`, `core/config.py`, `dag/graph.py`, `tools.py`,
   `state.py`, `errors.py`, plus the under-counted set (`cli.py`,
   `execution_repository.py`, `azure_batch.py`,
   `data_flow_repository.py`, `sessions/routes.py`). *(R3, R5.)*
5. **`axiom-system-architect:catalog-debt`** + SCC#4 decomposition
   decision. *(After R5 informs; R2.)*
6. **Doc-correctness pass** — resolves ARCHITECTURE.md drift, plugin
   counts, audit-table count, ADR index. *(R10.)*

R1 (ADR-010 dispatcher verification) was closed during quality assessment.

---

## §10 Provenance & confidence

Every load-bearing claim in `99-stitched-report.md` and
`05-quality-assessment.md` carries an explicit confidence rating
(High / Medium / Low) and a citation chain back to either:

- A deterministic oracle artefact (`temp/l3-import-graph.json`,
  `temp/tier-model-oracle.txt`)
- A CLAUDE.md / ARCHITECTURE.md / PLUGIN.md statement
- A code `file:line` reference

The synthesis's provenance ledger lives in `99-stitched-report.md` §10.
The assessment's confidence ledger lives in `05-quality-assessment.md` §8.

This is the discipline that separates a "professional architectural
documentation set" from a vibes-based architecture review. Every claim
is traceable; every gap is named; every recommendation is mapped to a
finding.
