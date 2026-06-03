# L2 engine/ Cluster Validation Report

**Date:** 2026-04-29
**Validator:** analysis-validator subagent
**Scope:** Δ L2-8 reduced-scope cluster contract (V1–V10)

## Verdict

**APPROVED**

All ten reduced-scope checks pass on substance. One minor numerical inconsistency exists between three documents (catalog vs. discovery+report) on the size of the `engine/__init__.py:__all__` list — the catalog is correct (25 names), discovery and report state 26. This does not change any structural conclusion and is recorded as a MINOR finding for cosmetic correction. No CRITICAL or WARNING issues; the cluster is structurally sound for synthesis.

## V1. Sub-subsystem inventory parity

**PASS.** `ls src/elspeth/engine/` yields exactly: `__init__.py`, `batch_adapter.py`, `bootstrap.py`, `clock.py`, `coalesce_executor.py`, `commencement.py`, `dag_navigator.py`, `dependency_resolver.py`, `executors/`, `orchestrator/`, `processor.py`, `retry.py`, `spans.py`, `tokens.py`, `triggers.py` (excluding `__pycache__/`). That is 13 standalone .py + 2 sub-packages = 15 entries, matching the catalog count exactly. All 15 catalog entries map 1:1 to a real directory or file — no inventions, no omissions. Sub-package file counts also verified: `executors/` has 16 .py files (catalog claims 16 ✓); `orchestrator/` has 7 .py files (catalog claims 7 ✓).

## V2. Oracle citation resolution

**PASS.** Spot-checked citations:

- `[ORACLE: temp/intra-cluster-edges.json stats.intra_edge_count = 0, intra_node_count = 0]` — verified in JSON (`stats.intra_edge_count: 0`, `stats.intra_node_count: 0`).
- `[ORACLE: temp/layer-check-engine-empty-allowlist.txt — 69 findings under R1/R4/R5/R6/R9]` — verified via `temp/layer-conformance-engine.json.defensive_pattern_findings_for_context_only.by_rule` (R1=6, R4=22, R5=37, R6=3, R9=1; total=69) and `temp/layer-check-engine.txt:3` (`VIOLATIONS FOUND: 69`).
- `processor.py:1533-1535` (retry classification) — read; lines 1532–1535 contain `def is_retryable(e: BaseException)` with `isinstance(e, PluginRetryableError)` at 1533 and `isinstance(e, ConnectionError | TimeoutError | OSError | CapacityError)` at 1535. Resolves.
- `declaration_dispatch.py:137,142` — read; line 137 is `except DeclarationContractViolation as exc:`, line 142 is `except PluginContractViolation as exc:`. Resolves.
- `tokens.py:19` — verified `from elspeth.core.landscape.data_flow_repository import DataFlowRepository` at exactly line 19.
- `orchestrator/__init__.py:1-22` — verified docstring contains "refactored from a single 3000+ line module into focused modules while preserving the public API" at lines 1–22.
- `coalesce_executor.py:1-5` — verified docstring contains "stateful barrier that holds tokens until merge conditions are met. Tokens are correlated by row_id".

## V3. Knowledge-map citation resolution

**PASS.** All 15 [DIVERGES FROM] markers and all sampled [CITES] markers resolve to entries in `00b-existing-knowledge-map.md`:

[CITES] sampled (8): KNOW-A25, KNOW-A26, KNOW-A28, KNOW-A44, KNOW-A45, KNOW-A46, KNOW-A47, KNOW-A48 — all resolve.
[CITES ADR-*] sampled: KNOW-ADR-009b, KNOW-ADR-010, KNOW-ADR-010e, KNOW-ADR-010f, KNOW-ADR-010i, KNOW-ADR-007/008/011/012/013/014/016/017 — all resolve.
[CITES] sampled (project): KNOW-A67, KNOW-A69, KNOW-A70, KNOW-C7, KNOW-C28, KNOW-C29, KNOW-C44, KNOW-C45, KNOW-P13, KNOW-P14 — all resolve.

[DIVERGES FROM] markers checked: KNOW-A70 (orchestrator LOC drift), KNOW-A26 (RowProcessor LOC growth), KNOW-A28 (CoalesceExecutor LOC growth), KNOW-C44 (test-path-integrity scope-conditional). The [DIVERGES FROM] markers are clearly justified inline (each cites verified evidence — file LOC counts or scope qualifications). No unresolved IDs found.

## V4. Cross-cluster boundary respect

**PASS.** Cross-cluster references in `04-cluster-report.md` fall exclusively into the three permitted categories:

- **Layer-relationship boilerplate**: "engine is L2 ... outbound edges confined to {contracts, core}" (executive synthesis).
- **Specific contracts/core file imports** (engine→core/contracts edges, layer-permitted): `tokens.py:19 → DataFlowRepository`, `executors/transform.py:16-23 → contracts.declaration_contracts`, `triggers.py:24 / commencement.py:12 / dependency_resolver.py:14 → core.expression_parser`, `bootstrap.py / dependency_resolver.py → contracts.pipeline_runner`.
- **Deferred bookmarks** in the explicit "Cross-cluster observations for synthesis" section (5 bookmarks, all one-line, all flagged as cross-cluster owner = synthesis pass).

No inline assertions about other clusters' internal structure. The §3 SCC analysis correctly mentions web/mcp/plugins/transforms/llm/telemetry/tui SCCs only to state they live "outside this cluster" — no claim about internals.

## V5. SCC handling

**PASS.** Both 02-cluster-catalog.md (Conventions §"Cluster-internal SCC handling") and 04-cluster-report.md (§"SCC analysis") explicitly record SCC = N/A with citation: `temp/intra-cluster-edges.json stats.sccs_touching_cluster = 0`. The report additionally clarifies why the JSON-isolated re-run is empty (engine is L2; oracle filters L3-only).

## V6. Layer-check oracle interpretation

**PASS.** All four documents consistently distinguish authoritative whole-tree-clean from engine-scoped corroboration:

- `00-cluster-coordination.md §Δ L2-6`: states whole-tree oracle is authoritative; engine-scoped JSON corroborates.
- `01-cluster-discovery.md §Layer conformance status`: explicitly explains exit-1 of `layer-check-engine.txt` is allowlist-prefix mismatch + 69 defensive-pattern findings, NOT layer violations.
- `02-cluster-catalog.md §Conventions`: identifies `layer-check-engine-empty-allowlist.txt` as the *defensive-pattern scanner*, not the layer-import enforcer.
- `04-cluster-report.md §Layer conformance verdict`: same framing; calls the empty-allowlist file findings "scoping artefact," not a conformance failure.

`temp/layer-conformance-engine.json` exists; `layer_import_violations_L1: []` and `type_checking_layer_warnings_TC: []` are both empty arrays, as required.

## V7. Depth-cap compliance

**PASS.** Spot-checked entries 3 (`processor.py`), 4 (`coalesce_executor.py`), and 1 (`engine/orchestrator/`) — all three L3-deep-dive flags. Each entry derives Responsibility from prior docs + first-30-lines docstring/imports only:

- Entry 3 cites `processor.py:1-9` docstring and `:21-26` import section. Concerns enumerate file:line locations from the defensive-pattern oracle (an external artefact), not from reading function bodies — which is permitted.
- Entry 4 cites `coalesce_executor.py:1-5` docstring (verified verbatim by validator) and `:7-30` import section.
- Entry 1 cites `orchestrator/__init__.py:1-22` (decomposition docstring), `:14-22` (module enumeration), and `:24-32` (re-exports). Concerns again use file:line locations from the oracle.

Per-entry word counts spot-checked (entries 3, 4, 1) are within the 300–500-word budget. No bodies of L3-flagged files were opened.

## V8. Closing sections present

**PASS.** `04-cluster-report.md` ends with the three required sections in correct order:

1. **Highest-confidence claims** — 3 numbered claims with substantive content (layer conformance; ADR-010 dispatch faithful + drift-resistant; terminal-state structural guarantee).
2. **Highest-uncertainty questions** — 3 numbered questions with substantive content (processor.py cohesion; declaration_dispatch.py R6 silent-except vs audit-complete; integration test-path integrity).
3. **Cross-cluster observations for synthesis** — 5 bulleted one-line items, each flagging a cross-cluster boundary.

## V9. Test-debt candidate count

**PASS.** Catalog Closing section "Test-debt candidates surfaced (per Δ L2-5)" lists exactly 3 items: (1) test-path integrity probe, (2) public-API surface test for `engine/__init__.py:__all__`, (3) declaration_dispatch.py R6 silent-except vs audit-complete. Item (3) appears as Q2 in the report's "Highest-uncertainty questions" — cross-reference satisfied. Item (1) appears as Q3. (Two of three test-debt candidates surface as uncertainty questions, exceeding the "at least one" bar.)

## V10. ADR-010 dispatch-site mapping

**PASS.** Catalog entry 2 (executors/) maps each dispatch site to specific adopter files with file:line citations: `pre_emission_check` ← `declared_required_fields.py:3-5`; `post_emission_check` + `batch_flush_check` ← `pass_through.py:9-13`, `declared_output_fields.py:3-6`, `can_drop_rows.py:3-6`, `schema_config_mode.py:3-6`; `boundary_check` ← `source_guaranteed_fields.py:3-5`, `sink_required_fields.py:3-5`. Spot-check verified:

- `declared_required_fields.py:1-7` — "registers for ONE dispatch site: pre_emission_check" ✓
- `pass_through.py:9-13` — "registers for TWO dispatch sites: post_emission_check, batch_flush_check" ✓
- `source_guaranteed_fields.py:1-7` — "registers for ONE dispatch site: boundary_check" ✓

The `03-cluster-diagrams.md` Component view (§2) reproduces the same mapping with a per-site → per-adopter directed graph and a cross-reference table.

## Issues found

**MINOR:**

1. **`__all__` count inconsistency.** The actual count is 25 (verified via `grep -c '^    "'` and direct read of lines 65–91). Catalog entry 15 says 25 (correct). `01-cluster-discovery.md:29` says 26 (wrong). `04-cluster-report.md:5` says 26 (wrong). Cosmetic; does not change any structural claim. Recommend a one-token correction sweep for "26-name" → "25-name" in discovery and report.

No CRITICAL or WARNING issues.

## Recommendation

**APPROVE.** Cluster is ready for the post-all-L2 synthesis pass. The MINOR `__all__` count drift can be corrected at synthesis time or left as a cosmetic erratum — it does not affect any analytical conclusion or downstream consumption of the cluster outputs.
