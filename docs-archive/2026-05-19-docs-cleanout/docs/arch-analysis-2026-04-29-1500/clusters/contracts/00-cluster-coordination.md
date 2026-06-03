# L2 #5 — `contracts/` cluster coordination

## Configuration

- **Cluster name:** `contracts` (l2-contracts-leaf-and-types per §7 Priority 5)
- **Cluster scope path:** `src/elspeth/contracts/`
- **Layer:** **L0** — leaf, ZERO outbound permitted [`enforce_tier_model.py:237` `"contracts": 0`]
- **Effort bracket:** Medium (2–3 hr; per §7 Priority 5 — UNCHANGED by §7.5)
- **Pass type:** Final L2 cluster pass; the cluster set is now complete for the post-L2 synthesis pass.
- **Depth cap:** Δ L2-3 — one entry per immediate subdirectory or coherent file group; NOT one entry per file. The cluster has 63 .py files (17,403 LOC); the catalog produces ~10–14 grouped entries.
- **Time constraint:** 2–3 hr target; >4.5 hr stop-and-report.
- **Complexity estimate:** Low — single layer, zero outbound, single >1,500-LOC file (`errors.py`), no SCC participation.

## Mandated reading order (executed)

1. `04-l1-summary.md §7` Priority 5 (`l2-contracts-leaf-and-types`) and §7.5 (Priority 5 unchanged from §7) ✓
2. `02-l1-subsystem-map.md §1` (`contracts/` entry, lines 14–24) ✓
3. `00b-existing-knowledge-map.md` — KNOW-A40, KNOW-A41, KNOW-A53, KNOW-A68, KNOW-C33, KNOW-C46, KNOW-C47, KNOW-C62, KNOW-C63, KNOW-C65, KNOW-C70, KNOW-ADR-006, KNOW-ADR-006a–d, KNOW-ADR-010, KNOW-ADR-010a, KNOW-ADR-010b, KNOW-ADR-010e, KNOW-ADR-010h ✓
4. `temp/l3-import-graph.json` — filtered to contracts/ scope (Δ L2-2; result: empty intra/outbound; see §Δ L2-2 below) ✓
5. `temp/tier-model-oracle.txt` — clean status quo; corroborates the L0 leaf invariant ✓
6. `clusters/engine/04-cluster-report.md` "Cross-cluster observations for synthesis" — 2 of 5 bookmarks land on contracts/ surfaces (`declaration_contracts`, `pipeline_runner`); both are exported per `__init__.py` and have direct catalog entries this pass ✓
7. ADR-006 narrative via `00b-existing-knowledge-map.md` KNOW-ADR-006a–d (Phase 1 ExpressionParser→core, Phase 2 contracts/hashing.py extraction, Phase 3 fingerprint+DSN, Phase 4 RuntimeServiceRateLimit, Phase 5 CI gate); ADR-010 narrative via KNOW-ADR-010, KNOW-ADR-010a/b/e/h ✓

## Non-negotiable constraints (per template Δ L2-1 through Δ L2-10)

- **Δ L2-1:** All artefacts under `clusters/contracts/`. No modification of L1 deliverables; no touching of other clusters.
- **Δ L2-2:** Filtered oracle JSON (`temp/intra-cluster-edges.json`) produced by Python filter, NOT hand-curated. Result is **empty** (intra=0, inbound=0, outbound=0) — this is architectural evidence of the L0 leaf invariant at the L3 graph resolution; the empty file is itself the finding.
- **Δ L2-3:** Sub-subsystem depth ONLY (one entry per immediate subdirectory or coherent file group). `errors.py` (1,566 LOC) is the only file >1,500-LOC threshold and is **L3 deep-dive candidate** — flagged, not opened.
- **Δ L2-4:** No cross-cluster claims. Edges from core/, engine/, plugins/, web/, mcp/, etc. into contracts/ are inbound-only and out of scope; cross-cluster observations land in §"Cross-cluster observations for synthesis" of the report.
- **Δ L2-5:** Tests live at `tests/unit/contracts/` (87 files) and `tests/integration/contracts/test_build_runtime_consistency.py` (1 file). Catalog cites tests as evidence for invariants where applicable; missing test coverage flagged as L2 debt candidate.
- **Δ L2-6:** Layer-check oracle invocations:
  - **Cluster-scoped:** `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth/contracts --allowlist config/cicd/enforce_tier_model` → exit 1, 225 R-rule defensive-pattern findings (R5=184, R2=20, R1=12, R6=9). These are NOT layer-import violations; they appear because the allowlist YAML keys are full-path prefixed against the whole `src/elspeth` tree and `--root src/elspeth/contracts` breaks the prefix match. Same artefact the engine cluster pass identified.
  - **Whole-tree authoritative:** `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` → exit 0, "No bug-hiding patterns detected. Check passed." Confirms L0 leaf invariant + R-rule allowlisting holds codebase-wide.
  - **dump-edges:** N/A — cluster is L0, not L3. Skipped per Δ L2-6 sub-clause.
  - Artefacts captured at `temp/layer-check-contracts.txt` (raw cluster-scoped run with allowlist), `temp/layer-check-contracts-empty-allowlist.txt` (cluster-scoped run with NO allowlist; identical 225-finding count, confirms allowlist-prefix-mismatch theory), `temp/layer-conformance-contracts.json` (structured summary).
- **Δ L2-7:** SCC handling **N/A**. Verified via `.venv/bin/python -c "import json; ..."` on `temp/l3-import-graph.json` `strongly_connected_components`: zero participating contracts/ nodes (the SCC list contains only L3 nodes — `web/*`, `plugins/transforms/llm/*`, etc.). Cluster is L0; SCC handling is structurally inapplicable.
- **Δ L2-8:** Validation subagent spawned after `02-cluster-catalog.md` and `04-cluster-report.md` are written. Cluster-scoped V1–V10 contract per the engine pass precedent.
- **Δ L2-9:** Time budget 2–3 hr; >4.5 hr stop-and-report.
- **Δ L2-10:** `04-cluster-report.md` ends with three required sections — Highest-confidence claims (top 3), Highest-uncertainty questions (top 3), Cross-cluster observations for synthesis.

## Execution log

- **2026-04-29 (start)** — Specialisation block proposed and approved by operator (this is the final L2 pass per operator note).
- **2026-04-29** — Workspace created at `clusters/contracts/temp/`. ✓
- **2026-04-29** — Δ L2-2 oracle filter executed via Python script; produced `temp/intra-cluster-edges.json` with stats `{intra_edge_count: 0, intra_node_count: 0, inbound_edge_count: 0, outbound_edge_count: 0, sccs_touching_cluster: 0}`. The empty result is architectural evidence: contracts/ has no L3 graph resolution edges (because contracts/ is L0 and the L3 graph is L3-only by construction); inbound count of 0 reflects that the L3 graph aggregates by L3 sub-package and contracts/ is L0, not an L3 destination. ✓
- **2026-04-29** — Δ L2-6 layer-check executed both scoped (exit 1, 225 R-rule findings) and whole-tree (exit 0, clean). Produced `temp/layer-check-contracts.txt`, `temp/layer-check-contracts-empty-allowlist.txt`, `temp/layer-conformance-contracts.json`. Whole-tree clean is the authoritative artefact for the L0 leaf invariant. ✓
- **2026-04-29** — SCC absence confirmed via `temp/l3-import-graph.json` grep. ✓
- **2026-04-29** — Sub-subsystem inventory: 63 .py files (17,403 LOC) under `src/elspeth/contracts/`; one sub-package (`config/`, 5 files, 1,231 LOC); one file >1,500-LOC threshold (`errors.py` 1,566 LOC, flagged as L3 deep-dive candidate). ✓
- **2026-04-29** — Discovery and catalog written. ✓
- **2026-04-29** — Diagrams and report written. ✓
- **2026-04-29** — Validation subagent spawned. Verdict: **APPROVED WITH AMENDMENTS** (V1–V10 PASS on substance; one WARNING on V3 KNOW-A* citation accuracy + two MINOR findings — Entry 14 file-enumeration completeness, Entry 6 word-count overrun). See `temp/validation-l2.md`.
- **2026-04-29** — Post-validation amendments applied:
  - **WARNING-V3 fixed.** Citation sweep against `00b-existing-knowledge-map.md`: 10 wrong KNOW-A* IDs replaced or removed across `02-cluster-catalog.md` and `04-cluster-report.md`. Replacements: KNOW-A1 (Entry 5) → KNOW-A17 + KNOW-A29 (Landscape facade + 4 repos); KNOW-A4 (Entry 5) → KNOW-A59 (Tier-1 audit trust); KNOW-A12 (Entry 8) → removed (no exact ID); KNOW-A14 (Entry 8) → KNOW-A55 + KNOW-A56 + KNOW-A57 + KNOW-A65 (schema validation lifecycle + ADR-003); KNOW-A20 (Entry 4) → KNOW-A69 (RFC 8785 implicit choice); KNOW-A22 (Entry 12) → KNOW-A20 (Checkpoint subsystem ~600 LOC, the actual content of A20); KNOW-A23 (Entries 7, 10) → KNOW-A59 (Tier-1 trust); KNOW-A24 (Entry 10) → KNOW-A60 (Tier-2 elevated trust); KNOW-A33 (Entry 2 inline + Report exec synthesis + Entry 2 closing) → KNOW-A11 inline / removed in synthesis / replaced with KNOW-C47 in closing; KNOW-A39 (Entries 8, 14) → divergence target re-pointed to `02-l1-subsystem-map.md §1` (Entry 8) / replaced with KNOW-C63 (Entry 14). KNOW-C42 (Entry 5 — wrong meaning) removed. The KNOW-C* and KNOW-ADR-* citations were verified accurate by the validator and are unchanged.
  - **MINOR-V1 fixed.** `guarantee_propagation.py` and `reorder_primitives.py` added to Entry 14's enumeration with one-line role descriptions.
  - **MINOR-V9 fixed.** Test-debt-to-Q mapping prose tightened in both `02-cluster-catalog.md §"Closing — test-debt candidates surfaced"` and `04-cluster-report.md §7`. Item 3 (test_errors) is now stated as the only 1:1 mapping (to Q2); Items 1 and 2 are stated as thematically supporting Q1 without being identical.
  - **MINOR-V7 (Entry 6 word-count overrun)** is left as-is per the validator's "content is dense rather than padded; substantively justified" framing. ~94 words over a soft 500-word budget on a triple-file entry covering a near-threshold L3-deep-dive flag is within the spirit of the depth cap.

## Cross-pass framing

The contracts cluster is the final L2 pass; the engine, core, composer, and plugins clusters precede it. This pass:

1. **Closes** L1 §7.5 still-open Q1 ("Responsibility cut between contracts/ (L0) and core/ (L1) post-ADR-006") at the **evidence** level — the catalog enumerates which contracts/ files were relocated by ADR-006 Phases 1–4 and surfaces concrete evidence on the L0/L1 boundary, **without proposing relocations**. Verdict-as-prescription is deferred to the architecture pack per Δ L2-7's "archaeology identifies, architecture prescribes" rule.
2. **Inventories** the freeze-contract surface area in contracts/ — 18 files use `freeze_fields(self, ...)` across 33 invocations. One CLAUDE.md "shallow-wrap" pattern detected at `config/runtime.py:338` (`MappingProxyType(dict(self.services))`) with the values being a frozen-dataclass type (`RuntimeServiceRateLimit` at `runtime.py:291` is `@dataclass(frozen=True, slots=True)`); per CLAUDE.md the shallow wrap is *acceptable* when values are guaranteed immutable, but the catalog records it as a place worth flagging because the deeper `freeze_fields` pattern would be more uniform.
3. **Maps** the L0 surface of the ADR-010 declaration-trust framework to the engine cluster's catalog: `audit_evidence.py` (AuditEvidenceBase ABC), `declaration_contracts.py` (4-site nominal-ABC framework), `tier_registry.py` (Tier-1 registry with module-prefix allowlist + freeze flag), `errors.py` (re-exports `FrameworkBugError` to break a circular import). The engine cluster's report flags two contracts-side surfaces explicitly; both are catalog entries this pass.
4. **Confirms** the L0 leaf invariant mechanically (whole-tree layer-check exit 0 + filtered-oracle empty-edge result). KNOW-A53 stands.

## Confidence statement

**High confidence** in catalog correctness for layer conformance, ADR-006 phase mapping, ADR-010 L0 surface, freeze inventory, and test coverage map. **Medium confidence** for the responsibility-cut analysis (Q1 evidence is presented; resolution is deferred). **Lowest analytical yield per LOC of any L2 cluster**, as predicted by §7 Priority 5; this pass is mainly a re-validation, not a discovery pass.
