# L2 #5 — `contracts/` cluster report

## Executive synthesis

The `contracts/` cluster is the **L0 leaf** of the ELSPETH 4-layer model: 63 Python files / 17,403 LOC under `src/elspeth/contracts/`, exposing a 208-name re-export surface from `__init__.py`, with **zero outbound runtime imports** to any other layer. Layer position is unambiguous (`enforce_tier_model.py:237` `"contracts": 0`); the whole-tree layer-check oracle ran exit 0 with "No bug-hiding patterns detected" — the **L0 leaf invariant is mechanically confirmed**. A single TYPE_CHECKING-only annotation ([Catalog Entry 9, `plugin_context.py:31`] `from elspeth.core.rate_limit import RateLimitRegistry`) is the cluster's only cross-layer reference; per `__init__.py:8-10` and CLAUDE.md it does not create runtime coupling.

Two ADRs structure the cluster: **ADR-006** ([CITES KNOW-ADR-006a–d], Accepted 2026-02-22) is the layer-remediation history — Phase 2's `contracts/hashing.py` extraction (visible at `hashing.py:8-11` as the cycle-breaking primitive) and Phase 4's `RuntimeServiceRateLimit` (visible at `config/runtime.py:291`) are the artefacts inside the cluster; Phases 1, 3, 5 produced effects outside or above. **ADR-010** ([CITES KNOW-ADR-010, KNOW-ADR-010a, b, e, h], Accepted 2026-04-19, amended 2026-04-20) is the declaration-trust framework whose **L0 surface lives in this cluster**: `audit_evidence.py` (AuditEvidenceBase nominal ABC, Decision 1), `tier_registry.py` (Tier-1 registry + `@tier_1_error` factory + freeze flag, Decision 2), `declaration_contracts.py` (4-site dispatcher framework + payload-schema H5, Decision 3); the engine-side dispatcher mechanics catalogued by the engine cluster (`engine/declaration_dispatch.py:1-26`) consume the L0 vocabulary defined here. Two of the engine cluster's five "Cross-cluster observations for synthesis" are **closed by this pass**: the contracts/declaration_contracts payload typedict surface and the `pipeline_runner` Protocol (Catalog Entries 6 and 13).

The cluster's defining structural commitment is the **post-ADR-006 boundary discipline**: the `__init__.py:79-87` and `:369-370` comment blocks explicitly state "Settings classes (RetrySettings, ElspethSettings, etc.) are NOT here — Import them from elspeth.core.config to avoid breaking the leaf boundary." This is institutional memory encoded in code; the comment pre-dates this pass and pre-empted exactly the temptation that would otherwise leak L1 (Pydantic Settings) into L0. The Settings ↔ Runtime alignment is mediated by the `config/` sub-package's `from_settings()` factories ([CITES KNOW-C33] — config-contracts verification script `python -m scripts.check_contracts` is the runtime check that this contract holds).

Beyond layer compliance, three cluster-internal patterns are load-bearing and worth surfacing:

1. **The freeze contract**: `contracts/freeze.py` (172 LOC) is the canonical implementation of `deep_freeze` (recursive frozen-container conversion), `freeze_fields` (frozen-dataclass `__post_init__` helper), and `require_int` (Tier-1 numeric validation that rejects `bool`). 18 contracts files use `freeze_fields(self, ...)` across **33 invocations** (verified by `grep`). The single non-uniform site is `config/runtime.py:338` (`MappingProxyType(dict(self.services))`); the values are a `@dataclass(frozen=True, slots=True)` (`RuntimeServiceRateLimit`), so the shallow wrap is acceptable per CLAUDE.md but recorded as the only place where pattern uniformity could improve.

2. **Cycle-breaking by extraction**: three intra-cluster cycles are explicitly broken by the cluster's structure — `errors.py ↔ tier_registry.py` (FrameworkBugError defined in tier_registry, decorated and re-exported from errors with explicit identity preservation per `errors.py:21-27`); `errors.py ↔ declaration_contracts.py` (early-defined types only); and `core.canonical ↔ contracts.hashing` (broken by ADR-006 Phase 2's extraction). Each break is documented in the file headers — institutional memory rather than convention.

3. **R-rule conformance under tier model**: 225 cluster-scoped R-rule findings (R5=184 isinstance, R2=20 getattr, R1=12 dict.get, R6=9 silent-except) are NOT layer violations and NOT cluster-specific debt. The whole-tree run is clean (exit 0); the cluster-scoped run produces 225 because the allowlist YAMLs in `config/cicd/enforce_tier_model/` use full-path keys against the whole `src/elspeth` tree, and `--root src/elspeth/contracts` breaks the prefix match. The empty-allowlist run produces the same 225 — corroborating the artefact theory. Of the 225, 184 land on `type_normalization.py` (the runtime-type normalization at the audit-record contract boundary, which is exactly where `isinstance` should be).

This pass closes [L1 §7.5 Q1](../../04-l1-summary.md) ("Responsibility cut between `contracts/` (L0) and `core/` (L1) post-ADR-006") **at the evidence level**. The catalog enumerates which contracts/ files were relocated by ADR-006 Phases 2 and 4 (visible artefacts inside the cluster), maps the post-ADR-006 boundary discipline (the `__init__.py` comment blocks and the `config/` sub-package's `from_settings()` factories), and identifies one Protocol-level smell (`plugin_context.py:31` TYPE_CHECKING import to `core.rate_limit`) where the boundary could be nudged toward Q1's "extract primitive" resolution. **Resolution is deferred**: per Δ L2-7 ("archaeology identifies, architecture prescribes"), this pass surfaces the question; the architecture-pack pass prescribes the cut.

## §1. Layer conformance verdict

**CONFIRMED.** The L0 leaf invariant holds at runtime. Evidence:

- `temp/intra-cluster-edges.json stats = {intra_edge_count: 0, intra_node_count: 0, inbound_edge_count: 0, outbound_edge_count: 0, sccs_touching_cluster: 0}` — the L3 oracle filtered to contracts/ scope is empty. (Note: contracts/ is L0 and the L3 oracle is L3-only by construction; the empty result is the *expected* architectural artefact and is itself the leaf-invariant evidence at this oracle's resolution.)
- `temp/layer-check-contracts.txt` — cluster-scoped run exits 1 with 225 R-rule findings; **no layer-import (L1) or TYPE_CHECKING-coupling (TC1) findings**. The 225 are an allowlist-prefix-mismatch artefact, corroborated by `temp/layer-check-contracts-empty-allowlist.txt` having the same 225-finding count.
- **Whole-tree run** (authoritative): exit 0, "No bug-hiding patterns detected. Check passed." — confirms both layer compliance and R-rule allowlist conformance codebase-wide.
- `temp/layer-conformance-contracts.json` records `layer_import_violations_L1: []` and `type_checking_layer_warnings_TC: []`.

The single cross-layer TYPE_CHECKING annotation (`plugin_context.py:31`) is permitted by `__init__.py:8-10` and is annotation-only; it does not create runtime coupling.

`dump-edges` step: **N/A — cluster is L0**, not L3. The dump-edges sub-clause of Δ L2-6 applies only to L3 clusters where intra-cluster edges exist in the L3 graph; for L0 the byte-equality assertion is structurally inapplicable.

## §2. KNOW-A53 / KNOW-C47 verdict

**RATIFIED.** "`contracts/` package has ZERO outbound dependencies, preventing circular imports and enabling independent testing" (KNOW-A53). The cluster's structure shows multiple deliberate mechanisms preserving this property:

- The **`__init__.py` exclusion of Settings** (lines 79–87 + 369–370): Settings classes live in `core.config`; `contracts/` re-exports their **Runtime protocols** but not the classes.
- The **`PluginConfigProtocol`** (`plugin_protocols.py:31-49`): defined in L0 specifically to avoid `contracts → plugins/infrastructure/config_base` dependency. Inverts the apparent contracts/plugins relationship.
- The **`PipelineRunner` Protocol** (`pipeline_runner.py:1-23`): defined in L0 so engine (L2) can accept a callback from cli/web (L3) without importing upward.
- The **`hashing.py` extraction** ([CITES KNOW-ADR-006b], `hashing.py:8-11`): "This module exists to break the circular dependency between contracts/ and core/canonical.py."

KNOW-A53 stands without divergence.

## §3. SCC analysis

**N/A.** `[ORACLE: temp/intra-cluster-edges.json stats.sccs_touching_cluster = 0]`. The L1 oracle's five SCCs are all L3 sub-package cycles (`web/*` 7-node, `mcp/*`, `plugins/transforms/llm/*`, `telemetry/*`, `tui/*`); contracts/ is L0 and is structurally outside the SCC analysis space. Δ L2-7 does not apply to this cluster.

The three intra-cluster cycles documented in §"Cycle-breaking by extraction" (Catalog Entry 7, Catalog Entry 4) are *broken* cycles — code structure with explicit cycle-breaking by extraction, not strongly-connected components in the import graph. They show up as acyclic edges in the C4 Component diagram (§2 of `03-cluster-diagrams.md`) because the breaking mechanism (forward declaration + decorated re-export, or extraction to a smaller module) eliminates the back-edge at the import resolution level.

## §4. ADR-006 evidence inventory (closes L1 Q1 at evidence level)

| ADR-006 phase | What it changed | What's visible inside `contracts/` (this pass) |
|---|---|---|
| Phase 1 — ExpressionParser → core/ | Removed expression-parsing primitives from `contracts/`; new home `core/expression_parser.py` | **Negative artefact**: no `expression_parser.py` in `contracts/` (verified by `ls`). The L1 entry's "Internal sub-areas" line does not list it. The engine cluster cites it imported from core at three sites (`triggers.py:24`, `commencement.py:12`, `dependency_resolver.py:14`). |
| Phase 2 — Extract `contracts/hashing.py` | Created the new module to break `contracts ↔ core/canonical` cycle | **Positive artefact**: `hashing.py` exists with explicit cycle-breaking rationale at `:8-11`; defines `CANONICAL_VERSION = "sha256-rfc8785-v1"` at `:25` as the single source of truth (consumed by `core/canonical.py` per the same comment). |
| Phase 3 — Fingerprint + DSN handling moved | Removed fingerprinting + DSN parsing from `contracts/` | **Negative artefact**: no `fingerprint.py` in `contracts/`. URL types (`url.py`, `database_url.py`) remain because they are *types*, not the *handlers* that did the work. |
| Phase 4 — `RuntimeServiceRateLimit` added | Created the per-service rate-limit dataclass | **Positive artefact**: `config/runtime.py:291` `@dataclass(frozen=True, slots=True) class RuntimeServiceRateLimit` exists with `from_settings()` factory; consumed by `RateLimitRegistry` (in core/) which is the TYPE_CHECKING-only annotation at `plugin_context.py:31`. |
| Phase 5 — CI gate enforcement | Added `enforce_tier_model.py` as the layer-import enforcer | **Positive artefact**: `enforce_tier_model.py:237` `LAYER_HIERARCHY = {"contracts": 0, ...}`; whole-tree exit 0 today. |

**Q1 evidence summary:** the post-ADR-006 boundary is materially clean. Two surfaces show *Q1-relevant softness*:

- **TYPE_CHECKING smell**: `plugin_context.py:31` annotates `RateLimitRegistry` (an L1 type) — the protocol-level interface is L0-shaped and could be extracted to `contracts.config.protocols` per ADR-006d's "Violation #11 Protocol".
- **`config/runtime.py:338` shallow MappingProxyType**: not strictly Q1 (this is a freeze-pattern uniformity question), but the relocation history puts `RuntimeServiceRateLimit` into the contracts cluster while the Pydantic side (`RateLimitSettings`) is in core; the asymmetric pattern at `:338` could be tidied for uniformity with the `freeze_fields` discipline elsewhere in the cluster.

**Resolution deferred** to the architecture pack per Δ L2-7. The L2 pass surfaces the evidence; the prescription is downstream.

## §5. Freeze contract surface inventory

Inventoried via `grep -lE "freeze_fields\(self" src/elspeth/contracts/*.py src/elspeth/contracts/config/*.py` (Catalog Entry 3 and Catalog Entry 14):

- **18 files** use `freeze_fields(self, ...)`.
- **33 invocations** total.
- **One non-uniform site**: `config/runtime.py:338` (`MappingProxyType(dict(self.services))`) — acceptable per CLAUDE.md (values are frozen dataclasses) but recorded as the only place where the cluster's pattern is non-uniform.
- **Zero forbidden anti-patterns**. Spot-checked via `grep` for the CLAUDE.md "Forbidden Anti-Patterns" list: `MappingProxyType(self.x)` (no copy) — none; `isinstance(self.x, dict)` as guard to skip — none; `isinstance(self.x, tuple)` to skip — none (the `diversion.py:66` `isinstance(self.diversions, tuple)` is offensive validation, not a skip-freezing guard).
- **CI enforcement**: `scripts/cicd/enforce_freeze_guards.py` (KNOW-C65) covers the surface.

The freeze surface is **clean and uniform** with the one cited exception. Tests for the canonical pattern: `tests/unit/contracts/test_freeze.py`, `test_freeze_regression.py` (the regression file's existence implies prior bug coverage; not opened at L2 depth).

## §6. ADR-010 L0 surface map

| ADR-010 Decision | Catalog entry / file | What's defined here | Engine cluster's consumer |
|---|---|---|---|
| Decision 1 — AuditEvidenceBase nominal ABC | Entry 6 / `audit_evidence.py` | The ABC + checked `__init__` wrapper closing the CPython 3.13 BaseException fast-path bypass (`audit_evidence.py:32-66`) | Used by every audit-bearing exception class (Tier-2 DTOs in `errors.py`, declaration violations) |
| Decision 2 — `@tier_1_error` registry | Entry 7 / `tier_registry.py` | Factory decorator with module-prefix allowlist (`elspeth.contracts.*`, `elspeth.engine.*`, `elspeth.core.*`); freeze flag at end of bootstrap | The bootstrap calls `freeze_tier_registry()`; the decorator is applied at `errors.py:24-27` to `FrameworkBugError` |
| Decision 3 — DeclarationContract framework | Entry 6 / `declaration_contracts.py` | 4 dispatch sites (DispatchSite StrEnum), nominal ABC `DeclarationContract`, `@implements_dispatch_site` decorator, bundle types per site, `DeclarationContractViolation`, sibling `AggregateDeclarationContractViolation`, registry mechanics | Engine cluster's `declaration_dispatch.py` (4 dispatch sites × 7 contract adopters) consumes this surface end-to-end |
| H5 — payload-schema deny-by-default | Entry 6 / `declaration_contracts.py:25-26` | Each violation declares `payload_schema`; payload validated at construction before deep-freeze | Engine cluster's adopter files (7 of them) declare per-contract payloads |
| Secret scrubbing | Entry 11 / `secret_scrub.py` | Last-line-of-defence redaction for `DeclarationContractViolation` payloads | Used by the dispatcher before audit-record write |

The L0/L2 split is clean: contracts defines the **vocabulary** (ABCs, Protocols, dataclass shapes, enums); engine defines the **mechanics** (dispatcher, registry-walk, audit-complete loop, raise-or-aggregate). The engine cluster catalog confirms 4 sites × 7 adopters; this cluster confirms the protocol surface those adopters and that dispatcher rely on.

## §7. Test coverage map (per Δ L2-5)

`tests/unit/contracts/` has 87 files; `tests/unit/contracts/config/` has 6 sub-files; `tests/integration/contracts/` has 1 file. Catalog entry coverage map is in `01-cluster-discovery.md §Test coverage map`. **Direct unit coverage exists for every catalog entry's central invariant** with three exceptions (test-debt candidates):

1. No `test_public_surface.py` — the 208-name `__all__` list has no stability test.
2. No `test_tier_registry_pytest_gating.py` — the pytest-allowance widening of the module-prefix allowlist has no robustness test.
3. No `test_errors.py` — `errors.py` (1,566 LOC, the only L3 deep-dive candidate) has only indirect coverage through violation-class tests.

Item 3 (test_errors absence) maps 1:1 to Q2 in §"Highest-uncertainty questions" (the `errors.py` split question). Items 1 (test_public_surface) and 2 (test_tier_registry_pytest_gating) are independent observations, thematically reinforcing Q1's post-ADR-006 boundary framing without being identical to it.

## §8. Cross-cluster handshake closure

The engine cluster's `04-cluster-report.md §"Cross-cluster observations for synthesis"` named **two contracts-side bookmarks** for this cluster to pick up:

| Engine cluster bookmark | This cluster's catalog entry | Disposition |
|---|---|---|
| `engine ↔ contracts.declaration_contracts (ADR-010 payloads)` — `executors/transform.py:16-23` imports `PostEmissionInputs`, `PreEmissionInputs`, `derive_effective_input_fields` | Catalog Entry 6 | **Closed** — Entry 6 enumerates the 4-site framework's bundle types; the L0/L2 split is clean. |
| `engine ↔ contracts.pipeline_runner protocol` — `bootstrap.py` and `dependency_resolver.py` consume `PipelineRunner` | Catalog Entry 13 | **Closed** — Entry 13 confirms the Protocol is L0-defined, single-method, structural; the inverted-dependency pattern is canonical. |

Both engine-side observations are answered without contradiction. Three other engine-side bookmarks (`engine ↔ core.landscape DataFlowRepository`, `engine ↔ core.expression_parser`, `engine ↔ tests/integration/engine/ absence`) involve `core/` or testing-tier territory that this pass does not own; they remain in the synthesis pass's queue.

## §9. Limitations of this pass (for the record)

- **`errors.py` body not opened.** L3 deep-dive deferred. The Tier-1 decoration site set is therefore not exhaustively pinned at this pass; coverage is via header inspection + tier_registry.py allowlist mechanism + indirect test exposure.
- **`declaration_contracts.py` body not opened.** Header (`:1-58`) was read in full and the framework architecture is explicit there; the per-site bundle dataclasses and the registry mechanics are inferred from the header narrative, the engine cluster's catalog of consumer sites, and the four dedicated bundle types named in `__init__.py`'s re-exports.
- **`audit.py` (922 LOC) and `schema.py` (851 LOC) bodies not opened.** Roles are read from headers; per-class invariants are inferred from `__init__.py` re-exports + downstream usage.
- **The schema-contract subsystem (Catalog Entry 8, ~3,500 LOC across 8 files) was not deep-dived.** It is internally cohesive and a future-L3-candidate per the Δ4 heuristic, but no single file >1,500 LOC threshold makes it not a current candidate.
- **The 225 R-rule findings in the cluster-scoped layer-check were not enumerated by file**; the head-of-file view (`temp/layer-check-contracts.txt:1-30`) plus the rule histogram (R5=184, R2=20, R1=12, R6=9) were the basis for the "allowlist-prefix-mismatch artefact" verdict, corroborated by the empty-allowlist run.

## L1 cross-references

This report supplements:

- `02-l1-subsystem-map.md §1 (contracts/)` — ratifies and refines the L1 entry. The L1 "Highest-risk concern" line specifically named `errors.py` (1,566 LOC) as the L2-deep-dive candidate; this pass honours that depth-cap (Catalog Entry 10).
- `04-l1-summary.md §7 Priority 5` — confirms the priority-5 ranking ("only one deep-dive candidate; responsibility well-cited from ADR-006 + multiple KNOW-A* claims; the L0 leaf invariant is mechanically enforced") is appropriate. The 2–3 hr effort bracket holds for this pass.
- `04-l1-summary.md §7.5 Still open Q1` — answered at evidence level: the catalog inventories ADR-006 phase artefacts, surfaces the single TYPE_CHECKING smell at `plugin_context.py:31`, and defers prescription per Δ L2-7.
- `04-l1-summary.md §7.5 Standing note for all L2 passes` — F5 (~97% unconditional runtime coupling) doesn't apply *within* the contracts cluster (zero outbound at runtime; one TYPE_CHECKING annotation = explicitly conditional). The standing note's expected behaviour ("don't hunt for hidden TYPE_CHECKING coupling") matches the cluster's reality.

---

## Highest-confidence claims

(The post-L2 synthesis pass may include these verbatim.)

1. **The L0 leaf invariant is mechanically confirmed.** `enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` exits 0 with "No bug-hiding patterns detected"; `temp/intra-cluster-edges.json` shows zero outbound edges; `temp/layer-conformance-contracts.json` records `layer_import_violations_L1: []` and `type_checking_layer_warnings_TC: []`. KNOW-A53 stands. The single TYPE_CHECKING annotation to `core/` (`plugin_context.py:31`) is permitted and annotation-only.

2. **ADR-006 phase artefacts are visible inside the cluster and the post-relocation boundary is materially clean.** Phase 2's `hashing.py` extraction (`:8-11`), Phase 4's `RuntimeServiceRateLimit` dataclass (`config/runtime.py:291`), and Phase 5's CI gate (`enforce_tier_model.py:237`) are all present. The `__init__.py:79-87` and `:369-370` comment blocks encode the post-ADR-006 boundary as institutional memory. KNOW-ADR-006a–d ratified.

3. **The ADR-010 declaration-trust framework's L0 surface is complete and the L0/L2 split is clean.** The contracts cluster defines the vocabulary (`AuditEvidenceBase` ABC, `@tier_1_error` decorator + frozen registry, `DeclarationContract` 4-site framework with bundle types and payload-schema H5, secret-scrub last-line-of-defence); the engine cluster's already-mapped 4-site dispatcher × 7 adopters consumes this vocabulary end-to-end. Two engine-cluster cross-cluster bookmarks are closed by this pass.

## Highest-uncertainty questions

(The post-L2 synthesis pass owns these.)

1. **Does the `plugin_context.py:31` TYPE_CHECKING import to `core.rate_limit.RateLimitRegistry` warrant an "extract primitive" resolution per ADR-006d's Violation #11 Protocol?** The annotation is structurally permitted but it's the only cross-layer reference in the cluster; an extracted `RateLimitRegistryProtocol` in `contracts.config.protocols` would eliminate the TYPE_CHECKING block and tighten the L0/L1 boundary. **Owner: architecture pack.** This is the strongest Q1 evidence the catalog surfaces.

2. **Should `errors.py` (1,566 LOC) be split, and if so, along which seam?** The file holds Tier-1 raiseable exceptions, Tier-2 frozen audit DTOs, structured-reason TypedDicts, and re-exported `FrameworkBugError`. The Tier-1/Tier-2 distinction is currently encoded by inline comments (`errors.py:34` `# TIER-2: Frozen audit DTO ...`); a CI-enforced split (e.g., `contracts/errors_tier1.py` vs `contracts/errors_dtos.py`) would mechanise the discipline. **Owner: architecture pack.** This is also the highest-priority L3 deep-dive candidate in the cluster.

3. **Should the schema-contract subsystem (Catalog Entry 8, 8 files / ~3,500 LOC) be promoted from "top-level files" to a `contracts/schema_contracts/` sub-package?** The internal cohesion is high (all 8 files reference `FieldContract` / `SchemaContract` / `PipelineRow`); the names don't make their layering self-evident; a sub-package would mirror the `config/` partition and clarify the cluster's internal structure. The L1 entry summarised the schema-contract surface as one item; at L2 depth it is clearly a sub-cluster. **Owner: architecture pack.** Not blocking; pure organisational hygiene.

## Cross-cluster observations for synthesis

(One-line each; the post-all-L2 synthesis pass owns cross-cluster claims, not this cluster.)

- **contracts ↔ core (TYPE_CHECKING smell at `plugin_context.py:31`):** the only cross-layer reference in the cluster; candidate for ADR-006d "Violation #11" remediation. Owner: synthesis + architecture pack.
- **contracts ↔ engine (ADR-010 dispatch surface):** L0 vocabulary is complete; engine-cluster catalog entry 2 enumerated the 4-site × 7-adopter mapping; the `pipeline_runner` Protocol bookmark is also closed. Owner: synthesis (already aligned).
- **contracts ↔ core/landscape (audit DTO surface):** `audit.py` (922 LOC, header-only at this pass) is the L0-side of the Landscape audit-trail row contract; the L1 core cluster pass owns the L1-side write/read mechanics. Owner: synthesis.
- **contracts ↔ core/checkpoint (checkpoint family):** four checkpoint-family dataclasses (Catalog Entry 12) are L0 DTOs persisted by the L1 checkpoint repository. Owner: synthesis.
- **`type_normalization.py` R5 findings — trust-boundary correctness:** 184 isinstance findings on this single file are at the runtime-type-normalization trust boundary; whole-tree allowlist accepts them. The synthesis pass should note that this file is the cluster's densest correctness surface and a likely candidate for any future R-rule policy review. Owner: synthesis.
