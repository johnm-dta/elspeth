# 00 — engine/ cluster coordination

## Cluster identity

- **Cluster name:** engine
- **Scope path:** `src/elspeth/engine/`
- **Layer:** L2
- **L1 dispatch reference:** `04-l1-summary.md §7 Priority 1` (effort bracket 3–5 hr, unchanged by §7.5)
- **Workspace opened:** 2026-04-29 03:31

## Pre-pass reads (L1 inputs as read-only context)

- `04-l1-summary.md` — §7 Priority 1 + §7.5 standing note ingested.
- `02-l1-subsystem-map.md §3` — engine/ entry treated as authoritative; this L2 catalog *supplements*, never contradicts.
- `00b-existing-knowledge-map.md` — KNOW-A15, A25–A28, A44–A48, A70 + KNOW-ADR-007/008/009/010/011/012/013/014/016/017 cited throughout.
- `temp/l3-import-graph.json` — used as source for the cluster filter.
- `temp/tier-model-oracle.txt` — layer-conformance status quo (whole-tree clean).

## Deliverables

| File | Lines | Status |
|------|------:|--------|
| `00-cluster-coordination.md` | (this file) | written |
| `01-cluster-discovery.md` | 86 | written (Wave 2 subagent) |
| `02-cluster-catalog.md` | 362 | written (Wave 2 subagent; advisor-corrected framing) |
| `03-cluster-diagrams.md` | (Container view + ADR-010 component view) | written (coordinator) |
| `04-cluster-report.md` | (synthesis + Δ L2-10 closing sections) | written (coordinator) |
| `temp/intra-cluster-edges.json` | filtered (empty by design — engine is L2) | written (Wave 1) |
| `temp/layer-check-engine.txt` | full scoped scanner output | written (Wave 1) |
| `temp/layer-check-engine-empty-allowlist.txt` | scoped scanner with empty allowlist | written (Wave 1) |
| `temp/layer-conformance-engine.json` | JSON-isolated layer-only verdict | written (Wave 1, post-correction) |
| `temp/validation-l2.md` | Δ L2-8 validation gate | pending |

## Strategy

Sequential with delegated parallelism — 4 waves:

| Wave | Tasks | Mode |
|------|-------|------|
| W1 | Workspace + filter oracle + layer-check + sub-area inventory | coordinator (mechanical) |
| W2 | Discovery + catalog | parallel subagents |
| W3 | Diagrams + report (synthesis from catalog) | coordinator |
| W4 | Validation gate | subagent |

## Δ L2-6 layer-conformance verdict

- **Whole-tree oracle (authoritative):** `temp/tier-model-oracle.txt` exit 0 ("Check passed").
- **Engine-scoped corroboration:** `temp/layer-conformance-engine.json` reports 0 rule-`L1` upward-import violations and 0 rule-`TC` TYPE_CHECKING layer warnings inside `engine/`. Defensive-pattern findings (69, all already allowlisted whole-tree) are recorded for context only.
- **Δ L2-6 dump-edges sub-clause:** N/A — engine is L2, not L3.

## Δ L2-2 oracle filter result

`temp/intra-cluster-edges.json` reports `intra_node_count = 0`, `intra_edge_count = 0`, `inbound_edge_count = 0`, `outbound_edge_count = 0`, `sccs_touching_cluster = 0`. This is **expected and correct** — engine is L2; the source oracle filters L3-only. Empty file is the evidence that engine has no L3 graph footprint.

## Execution log

- 2026-04-29 03:31 — Workspace created at `clusters/engine/temp/`.
- 2026-04-29 03:31 — Filter applied to `l3-import-graph.json`; produced `intra-cluster-edges.json` with all-zero stats (expected per Δ L2-2).
- 2026-04-29 03:31 — Layer check scoped to `--root src/elspeth/engine` produced `layer-check-engine.txt` (exit 1 due to allowlist-key prefix mismatch, NOT layer violation).
- 2026-04-29 03:31 — Re-ran with empty allowlist → `layer-check-engine-empty-allowlist.txt` (69 already-allowlisted defensive findings re-surfaced).
- 2026-04-29 03:31 — JSON-isolated layer-only re-run → `layer-conformance-engine.json` (0 L1, 0 TC). Authoritative for engine layer-conformance claim.
- 2026-04-29 03:31 — Sub-area inventory verified: 5 sub-areas (orchestrator/, executors/, plus three deep-dive standalones) + 10 small standalones = 36 files / 17,425 LOC. Three files >1,500 LOC flagged for L3 deep-dive (`orchestrator/core.py` 3,281, `processor.py` 2,700, `coalesce_executor.py` 1,603).
- 2026-04-29 03:31 — Wave 2 dispatched: discovery + catalog subagents in parallel.
- 2026-04-29 03:33 — Discovery returned (86 lines, LOC distribution + entry-point inventory + 6 open questions).
- 2026-04-29 03:39 — Catalog returned (362 lines, 15 entries, 31 [CITES] + 15 [DIVERGES FROM]). Catalog adopted advisor-corrected framing of `layer-check-engine-empty-allowlist.txt` as the *defensive-pattern scanner* (not a layer-only check); routed 69 findings into per-entry Concerns rather than dismissing them.
- 2026-04-29 03:42 — Wave 3 (coordinator-authored): `03-cluster-diagrams.md` (Container view + ADR-010 component view, 28 layer-enforced edges drawn, all citation-backed) + `04-cluster-report.md` (executive synthesis + 4-priority answers + 3 highest-confidence claims + 3 highest-uncertainty questions + 5 cross-cluster bookmarks).
- 2026-04-29 03:43 — Wave 4 (validation gate) pending.
- 2026-04-29 03:51 — Wave 4 closed. `temp/validation-l2.md` returned **APPROVED**: all 10 reduced-scope checks (V1–V10) pass on substance; 0 CRITICAL, 0 WARNING, 1 MINOR. The MINOR finding was cosmetic drift on the `__all__` size: catalog correctly says 25 names, but discovery, diagrams, and report each said "26". Verified actual count via AST: 25 names. Drift fixed in 4 sites (`01-cluster-discovery.md` line 29, `03-cluster-diagrams.md` lines 15 + 110, `04-cluster-report.md` line 5); recheck confirms all docs now consistent at 25.
- 2026-04-29 03:52 — L2 #1 cluster analysis complete. Within §7 Priority 1 effort bracket (3–5 hr); no scope drift. The headline finding (mixed essential-vs-accidental for the KNOW-A70 concentration; processor.py unknown at L2 → L3 deep-dive needed) refines L1's quality-risk verdict without contradicting it. Five cross-cluster bookmarks deferred for post-L2 synthesis.

## Cluster-priority answer summary (pre-validation)

| Priority | Verdict |
|----------|---------|
| 1. KNOW-A70 essential-vs-accidental | **Mixed**: orchestrator residual of active decomposition; coalesce essential; processor unknown at L2 (L3 follow-up needed). |
| 2. Token-identity locus | **Three-locus split**: engine/tokens.py façade + core.landscape persistence + processor/orchestrator call sites. |
| 3. Retry + terminal-state | **Two-locus + structural guarantee**: retry.py loop + processor.py classification + state_guard.py NodeStateGuard context-manager invariant. |
| 4. Test-path integrity | **Scope-conditional**: KNOW-C44 binds at integration; engine unit scope tolerates mocks. No `tests/integration/engine/` exists — cross-cluster audit deferred. |
