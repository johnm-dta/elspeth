# L1 Pass Validation Report

**Date:** 2026-04-29
**Validator:** analysis-validator subagent
**Scope:** Δ7-reduced L1 contract (V1–V7 only; per-file claims and component-level diagram details out of scope)

## Verdict

**APPROVED**

All seven reduced-scope checks pass. Inventory parity is exact (11 subsystems, prose "12" divergence explicitly recorded), the layer enforcer re-ran clean, dependency claims align with the layer model and the oracle artefact, composite/leaf classification is mechanically correct on the four spot-checked subsystems, every [DIVERGES FROM] marker resolves to a real claim in the knowledge map and a representative sample of [CITES] markers also resolved, all six required deferrals are present with rationale and downstream owner, the Container diagram shows exactly 11 nodes and 27 layer-enforced edges, and all 11 entries are under the 300-word Δ2 cap with no per-file walkthroughs. One MINOR cosmetic issue is noted but does not warrant revision.

## V1. Subsystem inventory parity

**PASS.** `ls src/elspeth/` returns: `cli.py, cli_helpers.py, cli_formatters.py, __init__.py, composer_mcp, contracts, core, engine, mcp, plugins, telemetry, testing, tui, web` (excluding `__pycache__` and `py.typed`). Reconciled against catalog: 10 directories + 4 root files aggregated as `cli (root files)` = 11 subsystems, exactly matching the catalog and the expected list. The "12 vs 11" prose divergence is explicitly recorded in `02-l1-subsystem-map.md` line 3 (`[Δ4 note]`) and `04-l1-summary.md` §1 line 18.

## V2. Dependency-claim consistency

**PASS.** Re-ran `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`:

```
No bug-hiding patterns detected. Check passed.
EXIT=0
```

Layer table at `enforce_tier_model.py:236–240` confirms `contracts=0, core=1, engine=2, everything-else=3` (oracle artefact lines 52–64 cite this faithfully). All catalog outbound claims are subsets of the layer-permitted set:

- `contracts/` outbound = ∅ ✓
- `core/` outbound = `{contracts}` ✓
- `engine/` outbound = `{contracts, core}` ✓
- All 8 L3 subsystems outbound ⊆ `{contracts, core, engine, *L3}` with L3↔L3 marked **deferred** ✓

Diagram edge-accounting table (`03-l1-context-diagram.md` §2) shows exactly **27** layer-enforced edges (1 L1→L0 + 2 L2→{L0,L1} + 8×3 L3→{L0,L1,L2}). Verified by counting `-->` lines in the Container view: **27**.

## V3. Composite/Leaf classification

**PASS.** Spot-checked 2 composites and 2 leaves; all four exact:

| Subsystem | Files (verified) | LOC (verified) | Catalog claim | Class | Trigger |
|-----------|-----------------:|---------------:|---------------|-------|---------|
| contracts | 63 | 17,403 | 63 / 17,403 | COMPOSITE | ≥10k LOC ∧ ≥20 files ✓ |
| engine    | 36 | 17,425 | 36 / 17,425 | COMPOSITE | ≥10k LOC ∧ ≥20 files ✓ |
| telemetry | 14 |  2,884 | 14 /  2,884 | LEAF      | below all 3 thresholds ✓ |
| composer_mcp |  3 |   824 |  3 /    824 | LEAF      | below all 3 thresholds ✓ |

Δ4 heuristic (≥4 sub-pkgs OR ≥10k LOC OR ≥20 files) mechanically applied per entry. Per-entry triggers are explicitly named.

## V4. Citation resolution

**PASS.**

[DIVERGES FROM] markers (all 6 required) — every one resolves to a real KNOW-* in `00b-existing-knowledge-map.md` AND the divergence rationale is consistent with the cited claim:

| Marker | Resolves? | Cited claim | Catalog rationale |
|--------|-----------|-------------|-------------------|
| KNOW-A15 | ✓ | "Engine is ~12,000 LOC" | "stale relative to verified 17.4k" |
| KNOW-A18 | ✓ | "Testing subsystem ~9,500 LOC, ChaosLLM/ChaosWeb/ChaosEngine" | "conflates `src/elspeth/testing/` with `tests/`" |
| KNOW-A19 | ✓ | "Telemetry ~1,200 LOC" | "stale vs verified 2,884" |
| KNOW-A22 | ✓ | "Core ~5,000 LOC" | "containerisation accounting choice; verified 20.8k" |
| KNOW-A23 | ✓ | "Contracts ~8,300 LOC" | "stale; verified 17,403" |
| KNOW-A72 | ✓ | "46 plugins (summary count)" | "tension flagged, not resolved" |

[CITES] sample (8 random) — all resolve:

- KNOW-A12 (CLI Typer ~2,200 LOC) ✓
- KNOW-A14 (MCP server ~3,600 LOC, read-only analysis) ✓
- KNOW-A26 (Orchestrator/RowProcessor LOC figures) ✓
- KNOW-A35 (25 plugins across 4 categories) ✓
- KNOW-A53 (contracts ZERO outbound dependencies) ✓
- KNOW-C35 (`elspeth-mcp` read-only audit DB tools) ✓
- KNOW-C38 (telemetry primacy: audit fires first) ✓
- KNOW-G6 (`uvicorn elspeth.web.app:create_app`) ✓
- KNOW-G7 (SPA served from `frontend/dist/` after API/WS routes) ✓
- KNOW-P22 (TWO-place plugin registration: hookimpl + `cli.py:TRANSFORM_PLUGINS`) ✓

No unresolved citations encountered.

## V5. Deferrals presence

**PASS.** `04-l1-summary.md` §6 contains a 7-row table covering exactly the required deferrals:

| Deferral | Rationale | Owner |
|----------|-----------|-------|
| `web/frontend/` | Python-lens archaeologist cannot map TSX | frontend-aware archaeologist |
| `tests/` | distinct deliverable; would break depth cap | future `05-test-architecture.md` pass |
| `examples/` | per-pipeline analysis is per-vertical | worked-examples curation pass |
| `scripts/` | only enforcer touched per Δ5 | CI/tooling audit pass |
| Files >1,500 LOC | depth cap | corresponding L2 cluster pass |
| L3 ↔ L3 graph | Δ5 grep ban | L2 enumeration cluster (§7 #6) |
| Doc-correctness | not an architecture task | separate doc-correctness pass |

Each row names rationale + downstream owner. ✓

## V6. Diagram parity

**PASS.** `03-l1-context-diagram.md` §2 Container view contains exactly 11 subsystem nodes (verified by `grep -cE '^\s+(contracts|core|engine|...)\["'` = 11). Edge count verified at 27 layer-enforced edges (1 + 2 + 24, exactly matching the formula in the edge-accounting table). Layer subgraphs `L0` / `L1_layer` / `L2_layer` / `L3_layer` partition cleanly: `contracts` in L0, `core` in L1, `engine` in L2, the remaining 8 in L3 — matches the layer-table in `enforce_tier_model.py:236–240`.

## V7. Depth-cap compliance

**PASS.** Word counts per catalog entry (excluding the Closing section):

| # | Subsystem | Words |
|--:|-----------|------:|
| 1 | contracts/ | 202 |
| 2 | core/ | 221 |
| 3 | engine/ | 195 |
| 4 | plugins/ | 208 |
| 5 | web/ | 260 |
| 6 | mcp/ | 208 |
| 7 | composer_mcp/ | 242 |
| 8 | telemetry/ | 178 |
| 9 | tui/ | 149 |
| 10 | testing/ | 208 |
| 11 | cli (root files) | 188 |

All ≤ 300 words (Δ2 cap). Maximum is `web/` at 260. No entry contains a per-file walkthrough — "Highest-risk concern" lines name files with LOC counts but do not open them. The Closing's "Files ≥1,500 LOC observed at L1" is a 12-row table with no analysis: file path, LOC, subsystem only. Confirmed list-not-analysis. ✓

## Issues found

**MINOR (cosmetic only — no revision required):**

1. **Line-number citation for the layer table is inconsistent across the three documents.** `02-l1-subsystem-map.md` cites `enforce_tier_model.py:52–64` (which actually corresponds to oracle artefact lines, not script lines), while `03-l1-context-diagram.md` cites `enforce_tier_model.py:237–247` (close to the actual script range of 236–248 covering both `LAYER_HIERARCHY` and `LAYER_NAMES`). The truth-source is unambiguous (`LAYER_HIERARCHY` at line 236, `LAYER_NAMES` at line 243), and the catalog's intent is correctly conveyed via the oracle artefact, but a future tidy-up could align all three citations to a single canonical form (`enforce_tier_model.py:236–248` or `temp/tier-model-oracle.txt:52–64`). Not a contract violation — both citations point a reader to the right place.

No CRITICAL or WARNING issues found.

## Recommendation

**APPROVE.** L1 pass is contract-compliant, internally consistent, and faithfully represents the verified state of the codebase. The single MINOR finding is cosmetic and does not block progression to L2 dispatch.
