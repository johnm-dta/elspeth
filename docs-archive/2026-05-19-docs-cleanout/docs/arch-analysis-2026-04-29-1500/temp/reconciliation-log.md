# Phase 8 Reconciliation Log

**Date:** 2026-04-29
**Scope:** Cross-cluster tension detection across the five L2 cluster reports (engine, core, composer, plugins, contracts), checked against the L1 dispatch queue (§7), Phase 0 amendments (§7.5 F1–F5), the L3 import graph oracle (`temp/l3-import-graph.json`), and the existing-knowledge map (`00b-existing-knowledge-map.md`).

## Scanning method

Per Δ 8-6, an entry is required when ANY of these trigger conditions is met:

1. Cluster A asserts X about a cross-cluster boundary; cluster B asserts ¬X about the same boundary.
2. Cluster A's cross-cluster observation conflicts with another cluster's catalog.
3. A cluster's claim contradicts an entry in `00b-existing-knowledge-map.md` that wasn't `[DIVERGES FROM]`'d.
4. A cluster's claim contradicts the Phase 0 oracle.
5. A cluster's claim contradicts §7.5 amendments.

I walked the 54-entry manifest claim-by-claim and cross-referenced each cross-cluster observation, each highest-confidence claim that names another cluster, and each [DIVERGES FROM] marker against the candidate counterparts in (a) every other cluster's manifest entries, (b) the L1 §7/§7.5 sections, (c) the oracle `stats.*`, `edges`, and `strongly_connected_components` fields, and (d) the 00b knowledge-map IDs cited in each cluster's report (KNOW-A*, KNOW-C*, KNOW-G*, KNOW-ADR-*, KNOW-P*).

## Verdict

**Zero (0) cross-cluster contradictions detected. Two (2) near-miss reconciliations recorded for transparency.**

The five L2 cluster reports are mutually consistent at the boundaries they describe. Where one cluster's analysis names another cluster's territory, the named cluster's catalog substantively confirms the cross-cluster framing (often citing the same evidence from the L3 oracle). All [DIVERGES FROM] markers (KNOW-A24, KNOW-A35, KNOW-A72) target documentation, not other cluster catalogs, and are correctly handled per the L1 dispatch instruction to flag-not-resolve.

This is a notable structural-quality finding in itself: five independently-produced cluster reports, working under cluster-isolation discipline (Δ L2-4), produced no boundary-claim contradictions when joined.

## Entries

### Reconciliation R1 — F1 framing depth (composer cluster enriches §7.5 F1)

**Conflicting claims:**

- **§7.5 F1** (L1 amendment, line 145): `composer_mcp/` is coupled to `web/composer/`, not an independent sibling. Implicit framing: the coupling is a single oracle edge of weight 12.
- **Composer cluster confidence claim 1** (manifest entry, source `clusters/composer/04-cluster-report.md §5 (item 1)`): "F1 stands at the symbol level (`composer_mcp/server.py:1–40` imports the state types directly); F1's 'thin transport' framing is correct but understates the structural role — `web/composer/` is the cluster's data backbone, not just an MCP target."

**Tension:** Strictly, this is enrichment, not contradiction. F1 is a structural observation at the L3 boundary; the composer cluster's L2 reading of the same boundary at symbol-level depth says F1's framing is correct but understates. The tension is one of scope, not of fact.

**Evidence available:**

- The L3 oracle records the edge: `[ORACLE: edges array; composer_mcp → web/composer, weight 12, type_checking_only=false, conditional=false; sample sites composer_mcp/server.py:28, :29]`.
- The composer cluster has direct symbol-level access (manifest entries [composer] [confidence] item 1, [composer] [cross-cluster] bullet 4) and is therefore the higher-evidence source for the depth claim.
- Plugins, engine, core, contracts clusters each independently corroborate F1's existence at their own boundaries (e.g. composer terminal-cluster status: composer cross-cluster bullet 4 + plugins cross-cluster bullet 1).

**Resolution:** **Uphold both.** F1 stands as the L1-level structural amendment; the composer cluster's enrichment stands as the L2-level depth claim. The synthesis report's §3.2 (Strongly-connected zones) and §3.5 (Plugin spine) carry both: F1 for the cross-cluster topology, composer's enrichment for the SCC's internal structure. Neither cluster catalog needs amending.

**Catalog amendment flag:** None.

### Reconciliation R2 — TYPE_CHECKING accounting methodology (contracts cluster vs F5)

**Conflicting claims:**

- **§7.5 F5** (L1 amendment, line 149): "~97% of L3 edges are unconditional runtime coupling (0 TYPE_CHECKING-only, 2 conditional out of 77)." Implicit framing: there are zero TYPE_CHECKING-guarded layer references in the codebase visible to the oracle.
- **Contracts cluster cross-cluster bullet 1** (manifest entry, source `clusters/contracts/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 1)`): "contracts ↔ core (TYPE_CHECKING smell at `plugin_context.py:31`): the only cross-layer reference in the cluster; candidate for ADR-006d 'Violation #11' remediation."

**Tension:** Strictly, this is methodological, not factual. F5 reports `stats.type_checking_edges = 0` from the oracle, which is the count of edges whose **only** mode of import is TYPE_CHECKING-guarded — i.e., edges visible to a static-analysis tool that respects `if TYPE_CHECKING:` blocks. The contracts cluster's catalog (manifest entry [contracts] [confidence] item 1) confirms the layer-check tool's `type_checking_layer_warnings_TC: []` (also empty), and the catalog's "TYPE_CHECKING smell at `plugin_context.py:31`" comes from direct source-file reading of the contracts cluster, not from a tool-flagged warning.

**Evidence available:**

- The oracle JSON's `stats.type_checking_edges` field: 0 (verified at `temp/l3-import-graph.json`).
- The contracts cluster's `temp/layer-conformance-contracts.json:type_checking_layer_warnings_TC: []` (verified by the contracts cluster validator, V1).
- The contracts cluster's catalog entry for `plugin_context.py:31` cites the structural pattern (the file imports `core.rate_limit.RateLimitRegistry` inside a `TYPE_CHECKING` block) directly, not as a tool-flagged warning.

**Resolution:** **Both true; clarify methodology.** The oracle's `type_checking_edges = 0` means: of the 77 runtime edges in the graph, none are TYPE_CHECKING-guarded only — i.e., the oracle excludes TYPE_CHECKING imports from edge enumeration entirely (consistent with extraction methodology). The contracts cluster's structural identification of `plugin_context.py:31` is therefore not visible to the oracle and is not double-counted. The L1 standing note ("L2 archaeologists do **NOT** need to invest effort hunting for TYPE_CHECKING-guarded coupling at the L3 boundary") is correctly bounded: the oracle does not see TYPE_CHECKING; cluster catalogs may, and should, surface them as cluster-internal structural observations.

The synthesis report's §3.1 (Coupling surfaces) carries this clarification: oracle edges and TYPE_CHECKING annotations are different visibility tiers; both stand.

**Catalog amendment flag:** None. Both catalogs are correct. The clarification belongs in the synthesis report's coupling-surface treatment.

## Already-resolved divergences (not reconciled here per Δ 8-6 criterion 3)

The following [DIVERGES FROM] markers in the cluster reports are properly handled at the cluster level (the divergence is flagged, not resolved, per the L1 dispatch instruction). They do **not** trigger reconciliation entries because Δ 8-6 criterion 3 specifies "an entry in `00b-existing-knowledge-map.md` that wasn't `[DIVERGES FROM]`'d" — these were. They flow through to §7 (architectural debt candidates) and §8 (open architectural questions) as outstanding doc-correctness items:

- **KNOW-A24 (20 vs 21 audit tables):** core cluster confirms 20; KNOW-A24 says 21. `[DIVERGES FROM KNOW-A24]` marker in core cluster catalog. Owner per core's Synthesis-2: doc-correctness pass.
- **KNOW-A35 (25 plugins) vs actual 29:** plugins cluster confirms 29; `[DIVERGES FROM KNOW-A35]` marker. Owner per L1 §7.5 still-open Q4: doc-correctness pass.
- **KNOW-A72 (46 plugins):** plugins cluster reports KNOW-A72's "46" remains unexplained; `[DIVERGES FROM KNOW-A72]` marker; owner per L1 §7.5 still-open Q4: doc-correctness pass.
- **`__all__` count drift in engine:** discovery and report state 26; catalog and validator confirm 25. Engine validator MINOR finding. Cosmetic; not propagated to synthesis.

## Already-aligned cross-cluster handshakes

The following are cross-cluster claims that one cluster makes about another's territory; the named cluster's catalog substantively confirms each. They are **not** tensions and do not trigger reconciliation entries. Recorded here for synthesis-§3 traceability:

| Originator | Subject | Confirmer |
|---|---|---|
| engine cross-cluster bullet 1 (engine ↔ core.landscape via `tokens.py:19` → `DataFlowRepository`) | DataFlowRepository ownership | core confidence 2 (Landscape facade owns 4 repositories incl. DataFlowRepository) |
| engine cross-cluster bullet 2 (engine ↔ contracts.declaration_contracts ADR-010 payloads) | ADR-010 vocabulary | contracts confidence 3 (L0 surface complete; engine consumes the 4-site framework) |
| engine cross-cluster bullet 3 (engine ↔ core.expression_parser at three sites) | expression evaluator location | core uncertainty 2 + manifest mention of `dependency_config.py:63 → ExpressionParser` (intra-core lazy import) |
| engine cross-cluster bullet 4 (engine ↔ contracts.pipeline_runner protocol) | PipelineRunner protocol | contracts cross-cluster bullet 2 (engine-cluster bookmark closed) |
| core Synthesis-1 (50+ contracts names imported by core) | contracts surface inventory | contracts confidence 1 (L0 leaf, mechanically clean); minor inventory completeness flagged by contracts validator (`guarantee_propagation.py`, `reorder_primitives.py` not enumerated in catalog Entry 14) |
| core Synthesis-3 (engine cluster will need core's outbound `contracts/` surface) | engine→core→contracts transitive | engine cross-cluster bullet 2 (already cites the path through core) |
| core Synthesis-4 (`core/secrets.py` ↔ `web/composer/`) | runtime secret-ref resolver | composer uncertainty 2 (composer cluster's secrets surface composition is L3 territory) — alignment, not contradiction |
| core Synthesis-5 (mcp/composer_mcp separation rationale) | structural separation | composer cross-cluster bullet 3 (F2 reaffirmed) |
| composer cross-cluster bullet 1 (no direct engine import; routes via `plugins/infrastructure/`) | plugin spine routing | plugins confidence 1 + F3 (spine pattern) |
| composer cross-cluster bullet 4 (composer is terminal cluster — 0 inbound) | composer = leaf | composer confidence 3 (oracle-confirmed `cross_cluster_inbound_edges = []`) |
| plugins cross-cluster bullet 1 (`web/composer → plugins/infrastructure (w=22)` heaviest inbound) | composer→plugins coupling | composer cross-cluster bullet 1 (composer routes through plugins/infrastructure) |
| contracts cross-cluster bullet 2 (ADR-010 dispatch surface; engine bookmarks closed) | ADR-010 alignment | engine confidence 2 (ADR-010 dispatch faithfully implemented) |
| contracts cross-cluster bullet 3 (contracts ↔ core/landscape audit DTO surface) | L0 audit DTO ownership | core confidence 2 (Landscape sub-area + 4 repositories pattern) |
| contracts cross-cluster bullet 4 (contracts ↔ core/checkpoint family) | L0 checkpoint DTOs | core's manifest references `Checkpoint` and checkpoint-family DTOs in Synthesis-1 inventory |

All 14 handshakes resolve cleanly. None require synthesis-pass intervention beyond surfacing them as the §3 coupling-surface evidence base.

## Caveat on contracts cluster KNOW-A* citation accuracy (propagation hazard)

The contracts cluster validator (`temp/validation-l2.md`) recorded a WARNING: 10 `[CITES KNOW-A*]` markers in `02-cluster-catalog.md` reference IDs whose actual content does not match the inline rationale (KNOW-A1, A4, A12, A14, A20, A22, A23, A24, A33, A39). The structural conclusions of the catalog stand; the citation IDs are editorially defective.

**Synthesis pass mitigation:** When citing contracts cluster claims in `99-stitched-report.md`, prefer `[CLUSTER:contracts §<section>]` over re-quoting the catalog's `[KNOW-A*]` references. This keeps the editorial defect quarantined upstream and prevents propagation into downstream pack inputs. The drafting subagent for the stitched report receives this constraint explicitly.

This caveat is not a reconciliation entry per Δ 8-6 criteria — it is a propagation-hazard note for the synthesis discipline.

## Summary

- 0 cross-cluster contradictions
- 2 reconciled near-misses (R1 F1 enrichment, R2 TYPE_CHECKING accounting)
- 4 already-resolved [DIVERGES FROM] divergences (handed to doc-correctness pass)
- 14 cross-cluster handshakes confirmed aligned
- 1 propagation hazard noted (contracts KNOW-A* citation editorial defect)

The five L2 cluster reports are structurally sound and mutually consistent. Synthesis may proceed.
