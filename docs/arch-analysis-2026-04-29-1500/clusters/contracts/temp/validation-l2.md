# L2 contracts/ Cluster Validation Report

**Date:** 2026-04-29
**Validator:** analysis-validator subagent
**Scope:** Δ L2-8 reduced-scope cluster contract (V1–V10), final L2 cluster pass

## Verdict

**APPROVED WITH AMENDMENTS**

All ten reduced-scope checks pass on substance. Three issues found:

1. **WARNING — V3 KNOW-* citation accuracy.** A non-trivial number of `[CITES KNOW-A*]` markers in `02-cluster-catalog.md` reference IDs whose actual content in `00b-existing-knowledge-map.md` does not match the cited rationale. The ID slots used are real and the citations resolve to real entries; what does not resolve is the *semantic match* between the cited content and the inline justification. This is an editorial-accuracy issue, not a fabrication. Six confirmed mismatches (KNOW-A1, KNOW-A4, KNOW-A12, KNOW-A14, KNOW-A20, KNOW-A22, KNOW-A23, KNOW-A24, KNOW-A33, KNOW-A39 used in non-matching contexts). The fix is a one-pass citation sweep against the knowledge-map; the structural conclusions of the catalog are unaffected.

2. **MINOR — V1 inventory completeness.** Two files exist in `src/elspeth/contracts/*.py` that are not enumerated by any catalog Entry: `guarantee_propagation.py` and `reorder_primitives.py`. Entry 14's "Approximately 30 top-level files" is loose enough to plausibly subsume them, and 58 top-level files / 5 config files / 63 total matches the discovery census exactly, but neither file appears in any explicit list. Cosmetic; recommend a one-line addition in Entry 14's enumeration.

3. **MINOR — V7 word-count budget.** Entry 6 (audit-evidence framework / `declaration_contracts.py`) is ~594 words, ~94 words over the 500-word soft budget. Entry 5 (386), Entry 10 (468) are within range. Entry 6 covers three files including a near-threshold L3-deep-dive flag, so the over-spend has substantive justification and content is dense rather than padded. Cosmetic.

No CRITICAL findings. No cross-cluster boundary violations. No L3-flagged file body opened. The cluster is structurally sound for synthesis.

## V1. Sub-subsystem inventory parity

**PASS (with MINOR).** `ls src/elspeth/contracts/*.py | wc -l` = 58 (matches discovery's "58 top-level modules"); `ls src/elspeth/contracts/config/*.py` = 5 (matches discovery's "5 files"). LOC verifications all match exactly:

| File | Catalog claim | `wc -l` actual |
|---|---|---|
| `errors.py` | 1,566 | 1,566 ✓ |
| `declaration_contracts.py` | 1,323 | 1,323 ✓ |
| `audit.py` | 922 | 922 ✓ |
| `schema.py` | 851 | 851 ✓ |
| `schema_contract.py` | 797 | 797 ✓ |
| `plugin_protocols.py` | 753 | 753 ✓ |
| `config/runtime.py` | 655 | 655 ✓ |

`__all__` re-export count: `grep -cE '^    "' src/elspeth/contracts/__init__.py` = **208** (catalog claim: 208 ✓).

Two files are not explicitly enumerated by any catalog Entry: `guarantee_propagation.py` and `reorder_primitives.py`. Entry 14's wording ("Approximately 30 top-level files ... plus a handful of smaller siblings") is loose enough to plausibly include them, but neither name appears in the explicit list. Recorded as MINOR.

## V2. Oracle citation resolution

**PASS.** All sampled citations resolve verbatim:

- `temp/intra-cluster-edges.json` — `stats.outbound_edge_count: 0` ✓; `stats.sccs_touching_cluster: 0` ✓ (all five `stats.*` fields are 0).
- `temp/layer-conformance-contracts.json` — `layer_import_violations_L1: []` ✓; `type_checking_layer_warnings_TC: []` ✓.
- `enforce_tier_model.py:237` — `"contracts": 0,  # L0 — leaf, imports nothing above` ✓.
- `hashing.py:8-11` — cycle-breaking comment exists verbatim: "This module exists to break the circular dependency between contracts/ and core/canonical.py." ✓.
- `hashing.py:25` — `CANONICAL_VERSION = "sha256-rfc8785-v1"` ✓.
- `__init__.py:79-87` — Settings-exclusion comment block exists verbatim ✓.
- `__init__.py:369-370` — second Settings-exclusion comment block at lines 369–370 ("NOTE: Settings classes ... are NOT here / Import them from elspeth.core.config to avoid breaking the leaf boundary") ✓.
- `audit_evidence.py:32-66` — BaseException fast-path bypass discussion (`:32-38`) and `__init_subclass__` checked-init pattern (`:53-62`) verified ✓.
- `tier_registry.py:8-11` — module-prefix allowlist ("only callers in `elspeth.contracts.*`, `elspeth.engine.*`, or `elspeth.core.*`") ✓.
- `errors.py:21-27` — re-export pattern with `@tier_1_error(reason="ADR-008: framework internal inconsistency — engine bug", caller_module=__name__)` ✓.
- `declaration_contracts.py:1-58` — 4-site framework header verified verbatim, including `DispatchSite` StrEnum, `@implements_dispatch_site`, the four bundle-type families, audit-complete dispatch semantics, sibling aggregate ✓.
- `config/runtime.py:291` — `RuntimeServiceRateLimit` is `@dataclass(frozen=True, slots=True)` ✓.
- `config/runtime.py:338` — `object.__setattr__(self, "services", MappingProxyType(dict(self.services)))` shallow-wrap ✓ (the catalog cites :338 for `MappingProxyType(dict(self.services))`; actual line 338 in the file is `object.__setattr__(...)` and line 336 is the comment "Freeze services mapping to prevent external mutation"; the wrap call is effectively at :338).
- `plugin_context.py:31` — `from elspeth.core.rate_limit import RateLimitRegistry` (within `if TYPE_CHECKING:` block) ✓.
- `plugin_protocols.py:31-49` — `PluginConfigProtocol` rationale comment ("Defined here in L0/contracts to avoid a structural dependency from contracts → plugins/infrastructure/config_base") ✓.
- `pipeline_runner.py:1-23` — inverted-dependency Protocol pattern verified verbatim ✓.
- `diversion.py:62-69` — offensive validation pattern verified (raises `PluginContractViolation` on type mismatch) ✓.

## V3. Knowledge-map citation resolution

**WARNING.** Sampled `[CITES KNOW-*]` markers all resolve to real ID slots in `00b-existing-knowledge-map.md`. The ADR-* family resolves cleanly: KNOW-ADR-006, ADR-006a, ADR-006b, ADR-006d, ADR-010, ADR-010a, ADR-010b, ADR-010e, ADR-010h all match the catalog's inline rationale.

The KNOW-A* family has substantive content mismatches:

| Cited as | Catalog rationale | Actual map content (`00b-existing-knowledge-map.md`) |
|---|---|---|
| KNOW-A12 | "PipelineRow" / "row['foo'] dual-access wrapper" (Entry 8) | "CLI is built on Typer, ~2,200 LOC" |
| KNOW-A14 | "Unified Schema Contracts" (Entry 8) | "MCP Server is ~3,600 LOC" |
| KNOW-A20 | "canonical-JSON family / hash version" (Entry 4) | "Checkpoint subsystem is ~600 LOC" |
| KNOW-A22 | "checkpoint contract — resume safety" (Entry 12) | "Core is ~5,000 LOC" |
| KNOW-A23 | "Tier-1 family" (Entry 7, Entry 10) | "Contracts is ~8,300 LOC" |
| KNOW-A24 | "Tier-2 audit DTOs" (Entry 10) | "Audit DB has 21 tables" |
| KNOW-A33 | "Pydantic Settings family in `core/`" (Entry 2, Report §) | "Composite PK `(node_id, run_id)` on `nodes` table" |
| KNOW-A39 | "broad freeze inventory across contracts" (Entry 14); divergence target for schema-contract sub-cluster (Entry 8) | "Audited clients are 4: AuditedHTTPClient, ..." |
| KNOW-A1 | "Landscape three-layer architecture" (Entry 5) | "ELSPETH is described as Auditable Sense/Decide/Act" |
| KNOW-A4 | "Audit DB read-time contract is crash on bad data, ZERO coercion" (Entry 5) | "Audit storage is SQLite/SQLCipher in dev" |

Citations that resolve correctly (sampled): KNOW-A40 (phase-typed protocols, Entry 9 ✓), KNOW-A41 (concrete `PluginContext` satisfies all 4, Entry 9 ✓), KNOW-A46 (TokenInfo, Entry 11 ✓), KNOW-A53 (zero-outbound leaf, Entry 1, Entry 13, §2 ✓), KNOW-C33 (config-contracts script, Entry 2 ✓), KNOW-C42 (not actually cited by contracts cluster — engine is correct), KNOW-C47 (4-layer model, Entry 1 ✓), KNOW-C62 (canonical `freeze_fields`, Entries 3, 14 ✓), KNOW-C63 (`deep_freeze` recursive, Entry 3 ✓), KNOW-C64 (forbidden anti-patterns, Entry 3 ✓), KNOW-C65 (CI freeze guards, Entry 3 ✓).

The explicit `[DIVERGES FROM KNOW-A39]` in Entry 8 is justified inline ("the existing-knowledge map summarises the schema-contract surface as one item, but at L2 depth it's clearly an internally cohesive 8-file sub-cluster"). However, KNOW-A39's actual content is "Audited clients are 4" — entirely unrelated to schema contracts. The divergence framing is internally consistent but the ID is the wrong target.

**Diagnosis:** the catalog appears to use KNOW-A* IDs from a *different* numbering scheme than the actual `00b-existing-knowledge-map.md`. The IDs cited for ADR/* and KNOW-C* and a subset of KNOW-A* (40, 41, 46, 53) match correctly. This points to either (a) drift between two iterations of the knowledge map, or (b) a transcription error during catalog authoring.

**Severity:** WARNING. The catalog's substantive analysis is unaffected — every cited claim has an inline rationale that stands on its own merits and is corroborated by file:line citations and ORACLE references. The citation table needs a one-pass repair against the current knowledge-map.

## V4. Cross-cluster boundary respect

**PASS.** All cross-cluster references in `04-cluster-report.md` and `02-cluster-catalog.md` fall into the four permitted categories:

- **(a) Layer-relationship boilerplate**: "L0 is leaf"; "outbound zero" — appropriately confined to layer-position context.
- **(b) Specific inbound-edge mentions**: `core/landscape/` writes audit DTOs (Entry 5), `core/config/` calls `from_settings()` (Entry 2), `engine/declaration_dispatch.py:1-26` consumes the L0 vocabulary (Entry 6, §6) — each cited as inbound-from-named-file, never as a verdict on internal mechanics.
- **(c) Engine-cluster bookmark closures**: Entry 6 explicitly closes the engine cluster's declaration_contracts bookmark; Entry 13 explicitly closes the pipeline_runner bookmark; both are flagged "Closed by this entry" with engine-cluster cross-reference (`§"Cross-cluster handshake closure"` in the report). Δ L2-8 V4 explicitly permits this.
- **(d) Deferred bookmarks**: Five bullets in `04-cluster-report.md §"Cross-cluster observations for synthesis"` (one each: contracts↔core TYPE_CHECKING smell, contracts↔engine ADR-010 dispatch, contracts↔core/landscape audit DTO, contracts↔core/checkpoint, type_normalization.py R5 trust boundary). Each is one-line and explicitly delegated to the synthesis pass.

No verdict on what core/, engine/, plugins/, web/, mcp/ internally do beyond cited inbound edges. The §3 SCC discussion correctly mentions web/*, plugins/transforms/llm/*, mcp/*, telemetry/*, tui/* SCCs only to state they are "L3-only by construction" — no internal claim.

## V5. SCC handling

**PASS.** `temp/intra-cluster-edges.json stats.sccs_touching_cluster: 0` ✓. Re-verified `temp/l3-import-graph.json strongly_connected_components`: five SCCs total, none of which contain any 'contracts' or 'contracts/*' node. Confirmed via `python -c "import json; ..."` listing the five SCCs:

1. `['mcp', 'mcp/analyzers']`
2. `['plugins/transforms/llm', 'plugins/transforms/llm/providers']`
3. `['telemetry', 'telemetry/exporters']`
4. `['tui', 'tui/screens', 'tui/widgets']`
5. `['web', 'web/auth', 'web/blobs', 'web/composer', 'web/execution', 'web/secrets', 'web/sessions']`

Both `02-cluster-catalog.md §Conventions` and `04-cluster-report.md §3 SCC analysis` correctly record N/A and explain why (cluster is L0; oracle is L3-only by construction). The `00-cluster-coordination.md §Δ L2-7` and `01-cluster-discovery.md §SCC status` also state N/A consistently. The report's §3 additionally distinguishes "broken cycles" (the documented intra-cluster cycle-breaking-by-extraction patterns at Entries 4 and 7) from "strongly-connected components in the import graph" — a valuable clarification that does not muddle the verdict.

## V6. Layer-check oracle interpretation

**PASS.** All four documents (00, 01, 02, 04) consistently distinguish:

- **Authoritative whole-tree clean**: `enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` exit 0, "No bug-hiding patterns detected. Check passed." Cited explicitly in `00-cluster-coordination.md §Δ L2-6` (whole-tree authoritative), `01-cluster-discovery.md §Layer conformance status`, `02-cluster-catalog.md §Conventions`, `04-cluster-report.md §1`.
- **Cluster-scoped 225 R-rule findings as allowlist-prefix-mismatch artefact**: identified consistently in all four documents. Discovery and report explain the mechanism (cluster-scoped run reads allowlist YAMLs whose key field is full-path-prefixed against the whole `src/elspeth` tree; `--root src/elspeth/contracts` reports relative paths and breaks the prefix match). Both note it is the *same artefact the engine cluster pass identified*.
- **Empty-allowlist run produces SAME 225 count**: verified directly in artefact files (`grep "VIOLATIONS FOUND" temp/layer-check-contracts*.txt` returns "VIOLATIONS FOUND: 225" for both files). Both files have the same 4,286-line size. The catalog and discovery cite this as the corroboration that the 225 are not real allowlist debt.

`temp/layer-conformance-contracts.json` exists with `layer_import_violations_L1: []` ✓ and `type_checking_layer_warnings_TC: []` ✓. The `whole_tree_verdict_authoritative` field is "CLEAN" and the message is the verbatim "No bug-hiding patterns detected. Check passed."

`temp/layer-check-contracts.txt` (cluster-scoped with allowlist) and `temp/layer-check-contracts-empty-allowlist.txt` (cluster-scoped without allowlist) both exist; both report 225 violations; both are 4,286 lines.

## V7. Depth-cap compliance

**PASS (with MINOR word-count overrun on Entry 6).** L3-deep-dive flag honour spot-checked on three entries:

- **Entry 10 (`errors.py` 1,566 LOC)**: NO body opened. Citations are confined to the header `:1-40` (lines 1, 5, 12, 13, 14, 16-27, 30-31, 34-40 are all in the header range). Concerns explicitly say "Body not opened (Δ L2-3 honoured)" and the deep-dive question set is presented as deferred to L3. ✓
- **Entry 6 (`declaration_contracts.py` 1,323 LOC, near-threshold)**: header `:1-58` cited; framework architecture is described from the header narrative (which is exceptionally detailed — the file's docstring runs the full first 58 lines and includes Public Surface, Audit-complete semantics, Registry sections); no per-class enumeration of bundle-input/output dataclass families opened from the body. ✓
- **Entry 5 (`audit.py` 922 LOC)**: header `:1-25` only; the named DTOs are inferred from `__init__.py:28-57` re-exports + header skim — explicitly stated in Concerns and Confidence ("body not opened ... 25 named DTOs are inferred from the `__init__.py` import list ... plus header skim, not from line-by-line read"). ✓

Per-entry word counts:

| Entry | Word count | Budget (300–500) |
|---|---|---|
| Entry 5 (`audit.py`) | 386 | within ✓ |
| Entry 6 (audit-evidence framework) | **594** | **+94 over** |
| Entry 10 (`errors.py`) | 468 | within ✓ |

Entry 6 exceeds budget by ~94 words. Entry 6 covers three files (one near-threshold, one near-mid-size, one tiny) and includes the L0/L2 cross-cluster mapping for ADR-010 — content density justifies the over-spend. Recorded as MINOR; no structural concern.

## V8. Closing sections present

**PASS.** `04-cluster-report.md` ends with the three required Δ L2-10 sections in correct order:

1. **Highest-confidence claims** (lines 132–141): three numbered claims, each substantive (1. L0 leaf invariant mechanically confirmed; 2. ADR-006 phase artefacts visible & post-relocation boundary clean; 3. ADR-010 declaration-trust framework's L0 surface complete & L0/L2 split clean).
2. **Highest-uncertainty questions** (lines 142–150): three numbered questions, each substantive (1. plugin_context.py:31 TYPE_CHECKING import warrant ADR-006d "extract primitive"?; 2. errors.py 1,566 LOC split seam?; 3. schema-contract subsystem promote to sub-package?). Each includes "Owner: architecture pack."
3. **Cross-cluster observations for synthesis** (lines 152–160): five bulleted one-line items (contracts↔core TYPE_CHECKING; contracts↔engine ADR-010 dispatch; contracts↔core/landscape audit DTO; contracts↔core/checkpoint; type_normalization.py R5 trust boundary). Each is one-line with an "Owner: synthesis" tag.

Order matches contract; substance is non-trivial.

## V9. Test-debt candidate count

**PASS.** `02-cluster-catalog.md §"Closing — test-debt candidates surfaced (per Δ L2-5)"` lists three concrete items:

1. `tests/unit/contracts/test_public_surface.py` does not exist (Entry 1).
2. `tests/unit/contracts/test_tier_registry_pytest_gating.py` does not exist (Entry 7).
3. `tests/unit/contracts/test_errors.py` does not exist (Entry 10).

Δ L2-5 minimum bar (≥1 candidate): exceeded (3).

Cross-reference to "Highest-uncertainty questions": Q2 ("Should `errors.py` (1,566 LOC) be split, and if so, along which seam?") directly references the test-debt item #3 (test_errors absence is part of the errors.py L3 deep-dive deferred work). Q1 (plugin_context.py:31 TYPE_CHECKING) and Q3 (schema-contract sub-package) do not directly trace back to the catalog's three test-debt candidates — they are independent observations.

The catalog's claim ("The first and third are also Q1-relevant: they would lock the L0 surface against drift while the architecture pack works through the post-ADR-006 boundary question") and the report's claim ("These three test-debt items are also surfaced as Q1/Q2 in §Highest-uncertainty questions") are slightly imprecise — only test-debt #3 (test_errors) maps directly to a Q (Q2). The "Q1-relevant" framing is a fairly loose use of "relevant" — the items relate to Q1 ("post-ADR-006 boundary") thematically, not as a 1:1 mapping. Substance is correct (catalog has 3, ≥1 maps to a Q); precision of the prose could be tightened. Within the V9 contract, the bar is met.

## V10. ADR-010 L0-surface mapping accuracy

**PASS.** Entry 6 + Report §6 ADR-010 L0-surface map verified by direct file inspection:

- **Decision 1 — AuditEvidenceBase ABC**: `audit_evidence.py` (75 LOC, full read). The ABC + checked `__init__` wrapper closing the CPython 3.13 BaseException fast-path bypass is verified at `:32-66` (`_raise_if_abstract` at 43–51; `__init_subclass__` at 53–62; direct `__init__` guard at 64–66). The "structural Protocol was rejected" rationale is at `:1-9` of the docstring. **Confirmed as the L0 surface for Decision 1.** ✓
- **Decision 2 — `@tier_1_error` registry**: `tier_registry.py:1-15` describes the three safety mechanisms verbatim ("(1) factory decorator requires `reason` kwarg; (2) module-prefix allowlist `elspeth.contracts.*`, `elspeth.engine.*`, `elspeth.core.*`, plus `tests.*` under pytest; (3) `freeze_tier_registry()` at end of bootstrap raises `FrameworkBugError`"). **Confirmed as documented.** ✓
- **Decision 3 — DeclarationContract framework**: `declaration_contracts.py:1-58` describes the 4-site framework (DispatchSite StrEnum at :8-9, `@implements_dispatch_site` at :9-13, the four bundle types at :20-22, `DeclarationContractViolation` at :25-26 with payload-schema H5, sibling `AggregateDeclarationContractViolation` at :27-31, audit-complete dispatch semantics at :35-43). The H5 payload-schema language ("Subclasses declare `payload_schema` (H5 Layer 1)") matches the report's framing. **Confirmed as the L0 surface for Decision 3 with audit-complete dispatch and payload-schema H5.** ✓

Engine-side mechanics (4 sites × 7 adopters): the catalog's claim that the engine cluster's `executors/` and `engine/declaration_dispatch.py` cover this is consistent with the engine cluster's catalog Entry 2 (per the engine validation report's V10 spot-check, which verified the per-site → per-adopter mapping). The L0/L2 split (contracts defines vocabulary; engine defines mechanics) is a clean, evidence-supported claim.

## Issues found

**WARNING:**

1. **V3 — KNOW-A* citation accuracy.** Confirmed 10 `[CITES KNOW-A*]` markers reference IDs whose actual content does not match the inline rationale: KNOW-A1, KNOW-A4, KNOW-A12, KNOW-A14, KNOW-A20, KNOW-A22, KNOW-A23, KNOW-A24, KNOW-A33, KNOW-A39. Plus `[DIVERGES FROM KNOW-A39]` in Entry 8 targets the wrong ID (real KNOW-A39 is "Audited clients are 4", not the schema-contract surface). The KNOW-C* and KNOW-ADR-* citations are accurate; the catalog's substantive content stands; the structural verdicts of the catalog and report are unaffected. **Recommend a one-pass citation sweep before the synthesis pass consumes these documents** — synthesis may include the report's claims verbatim, and the citation IDs would propagate.

**MINOR:**

1. **V1 — `guarantee_propagation.py` and `reorder_primitives.py` not explicitly enumerated.** Entry 14's "approximately 30 top-level files ... plus a handful of smaller siblings" wording subsumes them implicitly, but neither name appears in any explicit list. Total file count (58 top-level + 5 config = 63) matches discovery. Recommend adding both to Entry 14's enumeration.

2. **V7 — Entry 6 word-count overrun.** ~594 words against a 500-word soft budget; covers three files including a near-threshold L3-deep-dive flag. Content is dense; not padding. Cosmetic.

3. **V9 — test-debt-to-Q mapping prose imprecision.** The catalog's "first and third are also Q1-relevant" and the report's "These three test-debt items are also surfaced as Q1/Q2" claims are slightly imprecise — only test-debt #3 (test_errors) maps 1:1 to a Q (Q2). The substance is correct (≥1 candidate maps); the prose could be tightened.

## Recommendation

**Proceed to post-L2 synthesis.** The cluster's structural conclusions are sound, the L0 leaf invariant is mechanically confirmed, the ADR-006 and ADR-010 evidence inventories are complete, and the cross-cluster handshake closures (engine cluster bookmarks #1 and #2) are correctly executed. The WARNING on KNOW-A* citation accuracy is a non-load-bearing editorial defect — the inline rationale stands on its own and the resolved-cleanly KNOW-C* / KNOW-ADR-* / KNOW-A40 / KNOW-A41 / KNOW-A46 / KNOW-A53 citations cover the load-bearing structural claims. Synthesis may consume the catalog and report substantively while flagging the citation table as needing a one-pass repair before the documents are quoted in downstream artefacts.

The L1 §7.5 still-open Q1 ("Responsibility cut between contracts/ (L0) and core/ (L1) post-ADR-006") is closed at the evidence level by this pass. The Q1 evidence — `plugin_context.py:31` TYPE_CHECKING smell, `config/runtime.py:338` shallow MappingProxyType, ADR-006 phase-artefact inventory — is consistently surfaced and explicitly deferred to the architecture pack per Δ L2-7.
