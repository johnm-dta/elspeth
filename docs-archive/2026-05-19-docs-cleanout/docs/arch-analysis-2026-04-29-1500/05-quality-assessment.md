# 05 — Architecture Quality Assessment

**Source workspace:** `docs/arch-analysis-2026-04-29-1500/`
**Assessor:** axiom-system-architect / architecture-critic
**Assessed:** 2026-04-29 (live re-validation against `HEAD` performed during assessment)
**Methodology:** Evidence-based critique against synthesis output (`99-stitched-report.md`),
five cluster reports, deterministic L3 import oracle (`temp/l3-import-graph.json`),
and live diagnostics (`enforce_tier_model.py check`, fresh `dump-edges`,
`wc -l` of all Python sources, frontend / test tree probes).

> **Posture:** Professional means accurate. Stakeholders need truth, not comfort.
> Findings are rated by objective severity (Critical / High / Medium / Low),
> not by stakeholder preference. Strengths are reported only where the evidence
> warrants them, not as a sandwich for the bad news.

---

## §1 Executive summary

ELSPETH is a **structurally well-disciplined** ~121,408-LOC Python codebase
with a CI-enforced 4-layer import model that is currently honoured cleanly
across all 11 top-level subsystems. The audit-trail vocabulary (ADR-010
declaration-trust, terminal-state-per-token, trust-tier model) is encoded as
**mechanical invariants** — context-manager guards, AST-scanning drift tests,
allowlist-with-justification CI gates — rather than as conventions. That is
unusually strong for a codebase of this size and complexity.

It is also a codebase with **non-trivial unfinished structural work**:

- **13 files ≥1,500 LOC** concentrate ~22% of production Python in 0.6% of
  files (13 of 2,000+). The synthesis's §8 deferred-deep-dive list names only
  7 of these — the other 6 (`cli.py` 2,357, `execution_repository.py` 1,750,
  `azure_batch.py` 1,592, `data_flow_repository.py` 1,590, `sessions/routes.py`
  1,563, `coalesce_executor.py` 1,603) are flagged at L1 or L2 but not pulled
  forward into the synthesis's open-questions section. **The actual
  large-file footprint is larger than the synthesis presents.**
- A **7-node strongly-connected component** spans every `web/*` sub-package
  and is structurally load-bearing; decomposition is non-trivial and
  currently unowned.
- Two material qualitative areas — the **frontend** (~13k LOC of TS/React
  under `web/frontend/`) and the **test architecture** (~351k LOC across
  ~851 files) — are entirely outside the input set's scope. Any claim
  that this is a complete architecture assessment without those is wrong;
  this assessment names the gap rather than papering over it.

**Overall quality verdict:** Strong foundation, well-instrumented for
audit-grade integrity, with concentrated structural debt in three named
hotspots (engine LOC concentration, web SCC, large-file population) and
one **inventory completeness defect** (`web/sessions/routes.py` at 1,563
LOC was missed by the L1 12-file deferral list). Subsystem-level
findings: Critical 0, High 1, Medium 12, Low 7. Cross-cutting
concerns add 2 Medium (frontend + composer credential flow); 1 was
resolved in-pass (ADR-010 dispatcher verification).

**Recommendation:** The codebase is safe to continue building on. The
single architectural item that should land before the next major
addition to `web/` is the **SCC#4 decomposition decision** — the 7-node
`web/*` strongly-connected component is structurally load-bearing today,
and adding a new sub-package extends the cycle by default (R2). The
synthesis-flagged ADR-010 dispatcher "verification gap" (synthesis §5.2 /
§7.2) was resolved during this assessment by direct L3 read: the pattern
is verified correct, with 1,923 LOC of dedicated test coverage already
in place (see §3.1 E1).

---

## §2 Assessment basis

### §2.1 Input documents consumed

| Document | Words / Size | Used for |
|---|---:|---|
| `99-stitched-report.md` | ~6,933 words, 119 citations | System-level claims (§4–§8) |
| `99-cross-cluster-graph.md` | C4 Container + Component diagrams | Cross-cluster topology |
| `04-l1-summary.md` | L1 dispatch queue + Phase 0 amendments §7.5 | Subsystem inventory baseline |
| `02-l1-subsystem-map.md` | 11 catalog entries | Per-subsystem responsibility |
| `clusters/{engine,core,composer,plugins,contracts}/04-cluster-report.md` | 5 reports | Per-cluster confidence + uncertainty + cross-cluster bookmarks |
| `temp/l3-import-graph.json` | 1,773 lines, 33 nodes, 77 edges, 5 SCCs | Coupling oracle (cited by JSON path) |
| `00b-existing-knowledge-map.md` | ~250 KNOW-* claims | Institutional knowledge baseline |

The methodology in this workspace is **uncommonly rigorous**: every
high-confidence claim has a citation chain back to either (a) a deterministic
oracle artefact, (b) a CLAUDE.md / ARCHITECTURE.md statement, or (c) a code
file:line. The provenance ledger in `99-stitched-report.md` §10 grades
each claim's confidence with the sources cited. This assessment inherits and
relies on that discipline.

### §2.2 Live re-validation performed

To prevent assessing a stale snapshot, the following live checks were run
against `HEAD` during assessment (2026-04-29):

| Check | Result | Implication |
|---|---|---|
| `enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` | "No bug-hiding patterns detected. Check passed." | The synthesis's headline layer-conformance claim still holds |
| `enforce_tier_model.py dump-edges --no-timestamp` vs frozen `temp/l3-import-graph.json` | Identical except `generated_at` and `tool_version` metadata fields | The oracle is **deterministic and reproducible** modulo metadata; cited graph claims survive |
| `wc -l` all `.py` under `src/elspeth/` | 121,408 LOC | Matches synthesis "~121,392" (Δ < 0.02%) |
| LOC of every file the synthesis cites by line count | All 7 files match exactly (`tools.py` 3804, `core.py` 3281, `processor.py` 2700, `config.py` 2227, `dag/graph.py` 1968, `state.py` 1710, `errors.py` 1566) | Synthesis is faithful to the tree |
| `find src/elspeth -name "*.py" -exec wc -l {} \;` filtered ≥1,500 | **13 files**, not the 7 named in §8 | Synthesis under-reports LOC concentration — see §3.5 below |

### §2.3 Limitations of the input set (and therefore of this assessment)

The input set explicitly defers four areas. This assessment names them as
gaps; it does not cover them.

1. **Frontend (`src/elspeth/web/frontend/`, ~13k LOC TS/React).** No
   coverage. A FastAPI backend that serves a SPA is not architecturally
   complete without the SPA. **A frontend-aware archaeologist pass is a
   prerequisite for any "complete architecture" claim.**
2. **Test architecture (`tests/`, ~351k LOC across ~851 files).** No
   coverage. The `tests/` tree is roughly 2.9× the size of `src/`; its own
   architecture (fixture topology, integration-vs-unit split, KNOW-C44
   production-path conformance) is materially load-bearing for the audit
   guarantees this codebase makes.
3. **Examples (`examples/`, 36 pipelines).** Inventoried by count only.
   Whether the examples exercise the public API faithfully or have drifted
   from current contracts is unknown.
4. **Files ≥1,500 LOC.** Flagged but not deeply read at L2 depth; six
   files (named in synthesis §8) plus six additional files identified
   during this assessment's live re-validation (see §3.5).

**A "world-class" architecture document set must own these gaps openly,
not hide them.** The assessment treats them as named limitations and
recommends their resolution in §6.

---

## §3 Subsystem assessments

Quality scores are 1–5 against a fixed rubric:
**5** = production-grade discipline, structurally enforced; **4** =
strong, mechanically verified, with named work in flight; **3** = adequate
but with material structural debt; **2** = weak, requires architectural
intervention; **1** = unsafe.

### §3.1 `engine/` — score **4 / 5**

**Critical issues:** 0 · **High:** 0 · **Medium:** 2 · **Low:** 1

| # | Severity | Finding |
|---|---|---|
| E1 | **Low** | **ADR-010 dispatcher audit-completeness — VERIFIED during this assessment.** The cluster report rated `declaration_dispatch.py:137,142` (actual path: `engine/executors/declaration_dispatch.py`) as "the highest-stakes verification gap in the engine cluster" — flagged because L2 depth could not read the dispatcher body. **This assessment performed the L3 read.** Lines 131–172 confirm: (a) both `except DeclarationContractViolation` (137–141) and `except PluginContractViolation` (142–150) branches **append to the `violations` list**; neither branch swallows; (b) post-loop logic correctly distinguishes 0/1/N=2+ violations with reference-equality preservation at N=1 (the N6 regression invariant); (c) the docstring (lines 1–26) accurately describes the behaviour as audit-complete-with-aggregation per ADR-010 §Semantics; (d) `tests/unit/engine/test_declaration_dispatch.py` (642 LOC) + `tests/property/engine/test_declaration_dispatch_properties.py` (1,183 LOC) + `tests/integration/pipeline/orchestrator/test_declaration_contract_aggregate.py` (98 LOC) provide 1,923 LOC of dedicated coverage including N=1 reference-equality and N=2+ aggregation paths. **Evidence:** direct read of `src/elspeth/engine/executors/declaration_dispatch.py:120–172`; `wc -l` of the three test files. **Verdict:** the synthesis's S5.2 / S7.2 verification gap is **closed**. No remediation needed; the recommendation is to keep the existing test suite in CI gates (which it currently is). |
| E2 | **Medium** | **`processor.py` (2,700 LOC) cohesion is unverified at L2 depth.** The docstring (`processor.py:1-9`) claims one cohesive responsibility (RowProcessor end-to-end). The visible imports (DAG navigation, retry classification, terminal-state assignment, ADR-009b cross-check, batch error handling, quarantine routing) span 6+ concerns. **Evidence:** `clusters/engine/04-cluster-report.md` "Highest-uncertainty questions" item 1; KNOW-A70 quality risk. **Impact:** maintenance burden + onboarding friction; not a runtime risk. **Recommendation:** L3 deep-dive. The cluster report is correct that essential-vs-accidental can only be answered at L3 depth. **Effort: M.** |
| E3 | **Medium** | **No `tests/integration/engine/` directory.** Engine integration coverage exists *somewhere* in the integration tree but cannot be located within engine cluster scope. CLAUDE.md / KNOW-C44 require integration tests use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()` — the rule cannot be verified from inside engine. **Evidence:** `clusters/engine/04-cluster-report.md` "Cross-cluster observations" bullet 5. **Impact:** test-architecture audit gap; the production-path-rule guarantee is currently un-auditable. **Recommendation:** A cross-cluster integration-tier audit (distinct from any single L2 cluster pass). **Effort: M.** |

**Strengths (evidence-anchored):**

- **Terminal-state-per-token invariant is structurally guaranteed**, not
  conventional. `engine/executors/state_guard.py:NodeStateGuard` implements
  "every row reaches exactly one terminal state" as a context-manager
  pattern, locked by `test_state_guard_audit_evidence_discriminator.py`
  and `test_row_outcome.py`. Context-manager-as-invariant is genuinely
  good architectural practice for safety properties — the type system
  cooperates with the runtime to make the invariant non-bypassable.
  *(`clusters/engine/04-cluster-report.md` "Highest-confidence claims"
  item 3.)*
- **ADR-010 dispatch surface is drift-resistant by construction.** The
  4-site × 7-adopter mapping is locked by an AST-scanning unit test
  (`test_declaration_contract_bootstrap_drift.py`). Adding a new adopter
  without registering it fails CI. *(`clusters/engine/04-cluster-report.md`
  item 2.)*
- **`orchestrator/core.py` 3,281 LOC is the residual of an in-progress
  decomposition, not stagnant debt.** The orchestrator was previously a
  single 3,000-LOC module, now refactored into six focused siblings with
  stable public API. This is **active remediation visible in the tree**,
  not a cluster-flagged risk waiting for someone. The cluster report's
  refinement of KNOW-A70 to a "mixed essential-vs-accidental" verdict is
  more honest than the original L1 framing. *(`clusters/engine/04-cluster-report.md`
  "The 4 cluster priorities" item 1.)*

### §3.2 `core/` — score **4 / 5**

**Critical issues:** 0 · **High:** 0 · **Medium:** 3 · **Low:** 1

| # | Severity | Finding |
|---|---|---|
| C1 | **Medium** | **`core/config.py` (2,227 LOC) cohesion is unverified.** Single-file Pydantic settings holding 12+ child dataclasses across checkpoint, concurrency, database, landscape, payload-store, rate-limit, retry, secrets, sinks, sources, transforms. Pydantic settings concentrate for cross-validation reasons; whether 2,227 LOC is "appropriately concentrated" or "accreted by addition" is open. **Evidence:** `clusters/core/04-cluster-report.md` §4 item 1. **Impact:** onboarding friction; `core/config.py` is the load-bearing entry into runtime configuration and the largest single config file. **Recommendation:** L3 deep-dive paired with an architecture-pack proposal (split or keep). **Effort: M.** |
| C2 | **Medium** | **`core/dag/graph.py` (1,968 LOC) blast radius vs test-lock coverage is unknown.** `ExecutionGraph` is consumed by every executor in `engine/`, by `web/composer/_semantic_validator.py`, by `web/execution/validation.py`, by `core/checkpoint/{manager,compatibility}`, and indirectly by every plugin via the schema-contract validation flow. **Evidence:** `clusters/core/04-cluster-report.md` §4 item 3. **Impact:** any change to `ExecutionGraph` semantics is a system-wide change. The test files (`test_graph.py`, `test_graph_validation.py`) exist but their assertion density vs the file's behavioural surface is unverified. **Recommendation:** L3 deep-dive on the public-contract test surface; consider a pinned snapshot of `ExecutionGraph` semantic invariants. **Effort: M–L.** |
| C3 | **Medium** | **Audit table count divergence: 20 observed vs 21 documented.** The cluster pass observed 20 schema tables in `core/landscape/`; KNOW-A24 documents 21. **Evidence:** `clusters/core/04-cluster-report.md` `[DIVERGES FROM KNOW-A24]`. **Impact:** doc-correctness, not architecture. ARCHITECTURE.md is one major iteration behind on Landscape schema. **Recommendation:** Doc-correctness pass. **Effort: S.** |
| C4 | **Low** | **`core/secrets.py` (124 LOC, runtime resolver) lives at `core/` root while `core/security/{secret_loader,config_secrets}.py` (529 LOC combined) live in the subpackage.** Responsibility-cut question between L0 contracts and L1 core post-ADR-006. **Evidence:** `clusters/core/04-cluster-report.md` §4 item 2; synthesis §8.1. **Impact:** organisational hygiene only. **Recommendation:** Consider relocating `core/secrets.py` into `core/security/` for namespace consistency, or document the rationale for the split inline. **Effort: S.** |

**Strengths:**

- **Landscape facade pattern is real, not aspirational.** `landscape/__init__.py`
  re-exports exactly `RecorderFactory` and the four named repositories
  (`DataFlowRepository`, `ExecutionRepository`, `QueryRepository`,
  `RunLifecycleRepository`); repositories are *not* re-exported through
  `core/__init__.py`. Callers can only reach the audit DB through
  `RecorderFactory`. The encapsulation is mechanically enforceable, not
  documentary.
- **"Protocol-based no-op parity" is a deliberate offensive-programming
  discipline.** `EventBus`/`NullEventBus`, `RateLimiter`/`NoOpLimiter`
  pairs ensure callers never branch on `is None`; absent functionality is
  represented by an active no-op object that satisfies the protocol.
  This is the right pattern for L1 primitives and the cluster's
  identification of it as a deliberate idiom (rather than incidental
  duplication) is correct. *(`clusters/core/04-cluster-report.md` §3
  item 3.)*

### §3.3 `plugins/` — score **4 / 5**

**Critical issues:** 0 · **High:** 0 · **Medium:** 2 · **Low:** 2

| # | Severity | Finding |
|---|---|---|
| P1 | **Medium** | **`plugins/transforms/llm/azure_batch.py` (1,592 LOC) is unread at L2 depth.** Largest single plugin file; not in the synthesis's §8 deferred-deep-dive list despite being above the threshold. **Evidence:** live `wc -l`. **Impact:** the LLM batch path is a high-stakes plugin (financial cost, audit coverage, retry semantics) and its internal cohesion is un-assessed. **Recommendation:** L3 deep-dive; align effort estimate against `processor.py` and `tools.py` review. **Effort: M.** |
| P2 | **Medium** | **Trust-tier discipline is verbal/structural, not runtime-enforced.** Every source module repeats the "ONLY place coercion is allowed" contract; every sink module repeats the "wrong types = upstream bug = crash" contract; the discipline is encoded in the `allow_coercion` config flag. But cross-cluster invariant tests (e.g., a fixture that injects a transform that coerces and asserts the run fails) do not exist. **Evidence:** `clusters/plugins/04-cluster-report.md` §11 item 2. **Impact:** the contract is honoured today by author discipline; CI does not catch a violator. **Recommendation:** A property-based / fixture-based runtime probe paired with an architecture-pack decision on whether to mechanise the discipline. **Effort: M.** |
| P3 | **Low** | **Plugin-count drift in ARCHITECTURE.md (KNOW-A35 25 vs KNOW-A72 46 vs verified 29).** Four post-doc plugins were added without a doc update. **Evidence:** `clusters/plugins/04-cluster-report.md` §11 item 3. **Impact:** documentation correctness. **Recommendation:** Doc-correctness pass. **Effort: S.** |
| P4 | **Low** | **SCC #1 module-level cycle (`plugins/transforms/llm` ↔ `plugins/transforms/llm/providers`).** Provider-registry pattern with deferred runtime instantiation cited at `transform.py:9-13`. Module-level cycle is visible to the import system but runtime-decoupled. **Evidence:** `clusters/plugins/04-cluster-report.md` §10 item 3. **Impact:** none at runtime; visible in static analysis. **Recommendation:** Compare cost of moving shared types into `plugins/infrastructure/` versus leaving the cycle visible. **Effort: S.** |

**Strengths:**

- **`plugins/infrastructure/` is the structural spine and it is honoured
  consistently.** All 23 intra-cluster edges flow toward `infrastructure/`;
  `plugins/sinks → plugins/infrastructure` weight 45 is the heaviest
  single L3 edge in the codebase. Sinks/sources/transforms are clients
  of one another's infrastructure layer, not peers — the dependency
  shape matches the documented design.
- **Trust-tier discipline is documented identically in every leaf module.**
  Repetition is not a smell here; it is the protocol that prevents
  drift. New contributors writing a new source see the same "ONLY place
  coercion is allowed" notice as every existing source.

### §3.4 `web/` + `composer_mcp/` — score **3 / 5**

**Critical issues:** 0 · **High:** 1 · **Medium:** 3 · **Low:** 1

This cluster is the lowest-scoring not because the code is poor — by every
mechanical measure it is competent — but because of three structural
features that combine to make it the system's highest-risk
architectural surface.

| # | Severity | Finding |
|---|---|---|
| W1 | **High** | **7-node strongly-connected component spans `web ↔ web/auth ↔ web/blobs ↔ web/composer ↔ web/execution ↔ web/secrets ↔ web/sessions`.** No acyclic decomposition is possible within `web/`. The cycle is structurally load-bearing — it implements the FastAPI app-factory pattern, where `web/app.py:create_app()` imports every sub-package's router (wiring leg) and sub-packages reach back via `from elspeth.web.config import WebSettings` and `run_sync_in_worker` (shared-infrastructure leg). Both directions are intentional. **Evidence:** `temp/l3-import-graph.json` `strongly_connected_components[4]`; `clusters/composer/04-cluster-report.md` §5 item 2. **Impact:** any architectural change in `web/` must reason about all 7 sub-packages simultaneously. Adding a new sub-package extends the SCC by default. **Recommendation:** This is an architecture-pack decomposition decision, not a refactor task. The right shape is probably (a) extract a `web/_core/` containing `WebSettings` + `run_sync_in_worker` so sub-packages depend on `_core` rather than the namespace root, (b) make `web/app.py` the only place that imports sub-package routers. Until then, **freeze new sub-package additions to `web/`** unless they are explicitly architecture-reviewed. **Effort: L (5–8 hr architecture-pack pass + L–XL implementation).** |
| W2 | **Medium** | **`web/composer/tools.py` (3,804 LOC) and `web/composer/state.py` (1,710 LOC) are joint-largest concentration of composer logic in the tree.** Together they are 5,514 LOC — larger than any non-engine subsystem at L2 depth. **Evidence:** live `wc -l`; `clusters/composer/04-cluster-report.md` §5 item 1. **Impact:** maintenance burden; the composer state machine is the most architecturally-consequential surface in the system after the engine and any change has high blast-radius. **Recommendation:** Decomposition is paired with the SCC#4 decision (W1). Isolated decomposition without the SCC context risks producing a worse cycle. **Effort: L.** |
| W3 | **Medium** | **`web/sessions/routes.py` (1,563 LOC) was missed by the L1 deferral list entirely.** The L1 catalog listed 12 files >1,500 LOC; this file was not among them. **Evidence:** live `wc -l` (1,563 LOC); cross-check against `04-l1-summary.md` §6 deferred list. **Impact:** inventory completeness defect. The file exists, is large, and is unread at L1 *or* L2 depth. **Recommendation:** Add to the deep-dive backlog. Re-run the >1,500-LOC scan as part of any future L1 inventory pass. **Effort: S (catalog) + M (eventual deep-dive).** |
| W4 | **Medium** | **`composer_mcp/` is structurally a sibling of `web/composer/`, not of `mcp/`.** The L1 catalog framed `mcp/` and `composer_mcp/` as siblings; the L3 oracle records zero edges between them and a weight-12 edge from `composer_mcp → web/composer`. **Evidence:** `temp/l3-import-graph.json` edges; PHASE-0.5 §7.5 F1; `clusters/composer/04-cluster-report.md` §5 item 1. **Impact:** the L1 mental model was wrong; the L2 cluster pass corrected it. The current code is fine, but anyone reading the L1 layout in isolation would be misled. **Recommendation:** Either move `composer_mcp/` under `web/composer/` or document the structural relationship in ARCHITECTURE.md. **Effort: S (doc) or M (relocation).** |
| W5 | **Low** | **`web/execution → .` (root) edge weight 3 purpose unclear.** Could be a benign re-export of a public symbol from `elspeth/__init__.py`, or a deferred-import hack to bypass an explicit cluster dependency. **Evidence:** `temp/l3-import-graph.json`; synthesis §8.5. **Impact:** unknown. **Recommendation:** L3 deep-dive on the import line(s); takes minutes. **Effort: S.** |

**Strengths:**

- **The composer cluster is a structural leaf in the import graph.** Zero
  inbound edges from any other cluster; only the two console-script entry
  points (`elspeth-web`, `elspeth-composer`) consume it. Architectural
  changes to composer cannot break library callers elsewhere — a
  remarkably clean blast-radius property for a 22k-LOC subsystem.
  *(`temp/l3-import-graph.json` `cross_cluster_inbound_edges = []`;
  `clusters/composer/04-cluster-report.md` §5 item 3.)*

### §3.5 `contracts/` — score **5 / 5**

**Critical issues:** 0 · **High:** 0 · **Medium:** 2 · **Low:** 2

The L0 leaf is the system's strongest cluster. It is mechanically
verified to import nothing above (zero outbound edges); its responsibility
discipline is coherent; its CI gates are stable; and the cross-cluster
handshakes against engine, core, and plugins are aligned. The score of 5
reflects the discipline visible in the cluster, not the absence of
improvements.

| # | Severity | Finding |
|---|---|---|
| K1 | **Medium** | **`contracts/errors.py` (1,566 LOC) mixes Tier-1 raiseable exceptions, Tier-2 frozen audit DTOs, structured-reason TypedDicts, and re-exported `FrameworkBugError` in a single file.** The Tier-1 / Tier-2 distinction is currently encoded by inline comments, not by file split. **Evidence:** `clusters/contracts/04-cluster-report.md` "Highest-uncertainty questions" item 2. **Impact:** the discipline currently relies on convention; a CI-enforced split (e.g., `errors_tier1.py` vs `errors_dtos.py`) would mechanise it. **Recommendation:** Split when the file next needs material edits. Don't split-for-the-sake-of-splitting; pair with the next ADR that touches Tier-1 error definitions. **Effort: M.** |
| K2 | **Medium** | **`contracts/plugin_context.py:31` TYPE_CHECKING smell.** The only cross-layer reference in the cluster; ADR-006d Violation #11 candidate. An extracted `RateLimitRegistryProtocol` in `contracts.config.protocols` would eliminate the TYPE_CHECKING block. **Evidence:** `clusters/contracts/04-cluster-report.md` "Highest-uncertainty questions" item 1; KNOW-ADR-006d. **Impact:** annotation-only; the runtime is not coupled. But TYPE_CHECKING imports are the canonical marker of a deferred structural fix, and ADR-006d has a "never lazy-import" rule that this violates. **Recommendation:** Extract the protocol; the pattern is well-understood. **Effort: S.** |
| K3 | **Low** | **`contracts/schema_contract` sub-package promotion is non-blocking.** Catalog Entry 8 (8 files, ~3,500 LOC) has high internal cohesion; promoting to `contracts/schema_contracts/` would mirror the `config/` partition. **Evidence:** `clusters/contracts/04-cluster-report.md` "Highest-uncertainty questions" item 3. **Impact:** organisational hygiene. **Recommendation:** Defer until a near-term ADR motivates it. **Effort: S.** |
| K4 | **Low** | **Catalog citation editorial defect (10 KNOW-A* references).** Citation IDs resolve but inline rationales mismatch. **Evidence:** `temp/reconciliation-log.md` "Already-resolved divergences". **Impact:** doc-correctness. **Recommendation:** Doc-correctness pass. **Effort: S.** |

**Strengths:**

- **L0 leaf invariant is mechanically confirmed.** Zero outbound edges in
  `temp/l3-import-graph.json`; layer-conformance JSON empty for both L1
  upward-import and TYPE_CHECKING findings. The leaf is a leaf, verifiably.
- **ADR-010 declaration-trust framework's L0 surface is complete.**
  `AuditEvidenceBase` ABC, `@tier_1_error` decorator + frozen registry,
  `DeclarationContract` 4-site framework with bundle types and
  payload-schema H5 enforcement, secret-scrub last-line-of-defence — all
  present, all consumed by engine via the contracts-defined protocols.

---

## §4 Cross-cutting concerns

### §4.1 Security

The audit-trail and trust-tier surfaces are correctly identified by the
synthesis as "the highest-stakes security territory because the audit
trail is the legal record" (§9). At the architectural level, the
defence-in-depth is real:

- **Trust-tier topology is structural**, not aspirational: Tier 3
  (external) → sources coerce → Tier 2 (pipeline) → transforms/sinks
  pass through → Tier 1 (audit DB) crash-on-anomaly. The `enforce_tier_model.py`
  CI tool detects defensive patterns at trust boundaries and is honoured
  by the codebase today.
- **Audit-trail completeness is end-to-end.** Engine encodes
  terminal-state-per-token via `NodeStateGuard`; core's Landscape facade
  persists the 8 terminal/non-terminal states across (verified) 20 schema
  tables; contracts owns the L0 audit DTO vocabulary.
- **Secret-scrub is a last-line defence at the L0 boundary.** Encoded in
  `contracts/declaration_contracts.py` via the H5 payload-schema
  enforcement.

**However**, three security concerns are genuinely open:

| # | Severity | Concern |
|---|---|---|
| SEC1 | **Resolved** | The synthesis's flagged audit-completeness verification gap in `engine/executors/declaration_dispatch.py:137,142` was closed during this assessment via direct L3 read — the pattern is correct aggregation, with 1,923 LOC of dedicated test coverage. See §3.1 E1. *The synthesis correctly identified this as needing verification; this assessment performed the verification.* |
| SEC2 | **Medium** | Composer credential flow (`web/secrets/` has zero outbound edges to other clusters at package-collapse granularity). Composer/execution rely on LLM-provider credentials. Whether credentials flow via `WebSettings` injection at request time or via some other mechanism is L3 inspection territory. **The fact that the import graph cannot answer this is itself a red flag** — credential flow should be visible to architectural analysis, not hidden in DI plumbing. |
| SEC3 | **Medium** | The frontend (out of scope for this analysis) handles authentication tokens and session state. **No security analysis of the frontend has been performed.** A defence-in-depth claim about a browser-facing FastAPI-plus-SPA system is incomplete without a frontend-side review. |

**Recommended downstream pack:** `ordis-security-architect` for STRIDE
threat modelling against the trust-tier topology, ADR-010 dispatcher
audit-completeness verification, and the composer credential-flow question.
The synthesis correctly names this in §9.

### §4.2 Performance & operability

Out of scope for the input set. The synthesis does not mention performance
characteristics, throughput, latency budgets, or scaling profile. This is
a legitimate scope choice for an architecture archaeology pass — but it
means **performance assertions cannot be made from this workspace**.

A `processor.py` (2,700 LOC) handling per-row processing is a
hot-path candidate; whether it is in the actual hot path (vs. being
dominated by source I/O or LLM-provider latency) cannot be answered here.

**Recommendation:** Defer performance assessment to a separate
profiling-driven pass (`axiom-python-engineering:profile`) with
representative workloads.

### §4.3 Maintainability

| Dimension | Assessment | Evidence |
|---|---|---|
| **Layer model honoured** | Strong | CI-enforced clean today |
| **Per-file LOC discipline** | Mixed | 13 files ≥1,500 LOC; ~22% of LOC in 0.6% of files |
| **Test coverage of core invariants** | Likely strong, but inverse-pyramid risk unknown | `tests/` is 2.9× src/; no test-architecture pass performed |
| **Documentation drift** | Material | KNOW-A35 vs A72 plugin-count, 20-vs-21 audit-tables, ADR-007..017 unindexed in ARCHITECTURE.md |
| **Onboarding-readiness** | Medium | CLAUDE.md is excellent (load-bearing institutional memory); ARCHITECTURE.md is one major iteration behind |
| **Architectural decision records** | Good | ADRs 001..017 exist; index discipline has slipped |

The maintainability profile is the area where structural strength and
documentation drift diverge most. The **code** is well-disciplined; the
**institutional documentation about the code** has not kept pace and is
the main onboarding friction surface today.

### §4.4 Testability

Out of scope for this pass. Two things that can nevertheless be said
from the input set:

- **Engine integration testing has no in-cluster directory.** The KNOW-C44
  production-path rule cannot be verified from inside the engine cluster
  scope. This is recorded as S7.3 in the synthesis.
- **`tests/unit/engine/conftest.py:23` `MockCoalesceExecutor` has an
  explicit "Tests bypass the DAG builder" comment.** This is fine at unit
  scope but should be checked at integration scope.

**Recommendation:** A test-architecture pass is the highest-leverage
follow-up after the security pass. The 2.9× src/-to-tests ratio is
either remarkable test discipline or an inverted pyramid; the workspace
cannot tell which.

### §4.5 The frontend gap

`src/elspeth/web/frontend/` (~13k LOC TS/React) is an architectural
component of a FastAPI-plus-SPA system. It is **outside the input set's
scope by design** — a Python-lens archaeologist cannot map TSX usefully —
but it is **inside the architectural perimeter of any honest system
description**.

Specific consequences of the gap:

- The synthesis's "composer cluster has 0 inbound cross-cluster edges"
  finding is **structurally true at the Python-import level** but
  semantically incomplete — the frontend consumes the composer's HTTP/MCP
  surface, and that consumption is invisible to `enforce_tier_model.py`.
- Authentication / session-state flow on the SPA side is unanalysed.
- The frontend's coupling to backend API contracts is unverified; API
  contract drift is a known source of production-grade SPA failures.

**Recommendation:** A frontend-aware archaeologist pass (e.g.,
`lyra-site-designer` or a TS/React-specialised codebase explorer) is the
prerequisite for any "complete architecture" claim. The current
workspace's `temp/synthesis-input-manifest.md` correctly identifies this
as out-of-scope; the README should *visibly* acknowledge it as a named
gap rather than a deferral.

---

## §5 Findings the input set under-counted

This section catalogues defects that this assessment surfaced during live
re-validation but that the synthesis does not currently name as findings.
They are not defects in the synthesis's *cited* claims — those held up
under scrutiny — but defects in the synthesis's *coverage*.

### §5.1 LOC-concentration under-count (inventory completeness)

The synthesis §8 lists 7 files "remain unread" after L2:
`tools.py` 3804, `state.py` 1710, `config.py` 2227, `dag/graph.py` 1968,
`processor.py` 2700, `errors.py` 1566, `orchestrator/core.py` 3281.

The live tree contains **13 files ≥1,500 LOC**:

```
3804  src/elspeth/web/composer/tools.py                       [§8]
3281  src/elspeth/engine/orchestrator/core.py                 [§8]
2700  src/elspeth/engine/processor.py                         [§8]
2357  src/elspeth/cli.py                                      [missing]
2227  src/elspeth/core/config.py                              [§8]
1968  src/elspeth/core/dag/graph.py                           [§8]
1750  src/elspeth/core/landscape/execution_repository.py      [missing]
1710  src/elspeth/web/composer/state.py                       [§8]
1603  src/elspeth/engine/coalesce_executor.py                 [resolved L2]
1592  src/elspeth/plugins/transforms/llm/azure_batch.py       [missing]
1590  src/elspeth/core/landscape/data_flow_repository.py      [missing]
1566  src/elspeth/contracts/errors.py                         [§8]
1563  src/elspeth/web/sessions/routes.py                      [MISSED AT L1]
```

Five files (`cli.py`, `execution_repository.py`, `azure_batch.py`,
`data_flow_repository.py`) appear in the L1 deferral list but were not
pulled forward into the synthesis's open-questions section, so the
"remain unread" claim under-counts. One file (`web/sessions/routes.py`)
was missed at L1 entirely — an inventory completeness defect.

**Severity: Medium** (process defect, not architectural).
**Recommendation:** §6 R3.

### §5.2 The synthesis does not assess what it correctly defers

The synthesis correctly delegates "decomposition decisions" to the
architecture-pack pass. But the workspace as currently shaped has no
evidence that the architecture pack has *been called*. Until it is, the
findings sit in §7 (debt candidates) without owners or remediation
trajectories.

This assessment **is** an architecture-pack call. The recommendations in
§6 are the architecture pack's verdict on the synthesis's debt
candidates.

### §5.3 The `coalesce_executor.py` path mismatch in cluster-report

The engine cluster report at `04-cluster-report.md:5` says
"`coalesce_executor.py` 1,603" and treats it as part of the engine
cluster. The file lives at `src/elspeth/engine/coalesce_executor.py`
(verified live), not `src/elspeth/engine/executors/coalesce_executor.py`.
The cluster report is correct; the synthesis is correct; this is a
nit-level observation. Recorded for completeness because architecture
documents that cite paths must cite the right paths.

---

## §6 Priority recommendations

Severity is the **architectural impact**, not the **effort to remediate**.
A High-severity finding may be S to remediate; a Low may be L. Both
matter.

### R1 — **Resolved during this pass** · ADR-010 dispatcher audit-completeness · No further effort

The synthesis's S5.2 / S7.2 "highest-stakes verification gap" was closed
during this assessment via direct L3 read of
`src/elspeth/engine/executors/declaration_dispatch.py:120–172`. Both
except branches append to the violations list; the post-loop logic is
correct for 0/1/N≥2 cases; 1,923 LOC of dedicated test coverage exists
across three test files (unit / property / integration). **Action:**
record the verification outcome in the synthesis (or append a footnote);
no architectural change required. Map to §3.1 E1.

### R2 — **High** · Architecture-pack decision on SCC#4 (`web/*` 7-node SCC) · Effort: **L**

The 7-node SCC is structurally load-bearing today. Adding any new
sub-package to `web/` extends the cycle by default. The right shape is
probably (a) extract `web/_core/` containing `WebSettings` and
`run_sync_in_worker`, (b) make `web/app.py` the only place that imports
sub-package routers. Until this is decided, **freeze new sub-package
additions to `web/` unless explicitly architecture-reviewed.** Map to
§3.4 W1 and synthesis §7.6.

### R3 — **Medium** · Re-run ≥1,500-LOC inventory + add missing entries to backlog · Effort: **S**

The L1 12-file deferral list missed `web/sessions/routes.py` (1,563 LOC).
The synthesis §8 deferred-deep-dive list under-counts the live ≥1,500-LOC
population by 5 files. Re-run the scan, update the L2 dispatch backlog,
schedule the deep-dives. The mechanical cost is minutes; the
documentation correctness payoff is material.

### R4 — **Medium** · Cross-cluster integration-tier audit · Effort: **M**

There is no `tests/integration/engine/` directory; KNOW-C44's
production-path rule is currently un-auditable from inside any single
cluster. Locate the integration-tier coverage of engine paths, verify
that `ExecutionGraph.from_plugin_instances()` and
`instantiate_plugins_from_config()` are used consistently, document the
test-architecture topology. Map to §3.1 E3 and synthesis §7.3 / §5.3.

### R5 — **Medium** · L3 deep-dive on `processor.py` and `core/config.py` · Effort: **M each**

These are the two largest files where essential-vs-accidental cohesion
is open. Without L3 reads, the architecture pack cannot make split-or-keep
recommendations. The reads are bounded effort; the verdicts unblock
downstream decisions. Map to §3.1 E2, §3.2 C1, synthesis §5.1, §5.4.

### R6 — **Medium** · Frontend-aware archaeologist pass · Effort: **L**

The ~13k-LOC TS/React frontend is materially load-bearing for security,
session state, and API-contract integrity claims. A Python-lens
archaeologist cannot cover it. Engage `lyra-site-designer` or a
TS/React-specialised explorer. Map to §4.5 and synthesis §9.

### R7 — **Medium** · Test-architecture pass · Effort: **L**

`tests/` is 2.9× the size of `src/`. Whether this represents remarkable
test discipline or an inverted pyramid is unknown. A
`ordis-quality-engineering:analyze-pyramid` pass (or equivalent) would
answer it. Map to §4.4.

### R8 — **Medium** · STRIDE threat model + audit-completeness verification · Effort: **L**

The synthesis correctly nominates `ordis-security-architect` in §9. The
inputs (trust-tier topology in §3.3, audit-trail completeness in §3.4,
ADR-010 dispatcher state in R1) are now ready. Engage the security pack
once R1 is closed. Map to §4.1.

### R9 — **Medium** · Document the composer credential flow · Effort: **S**

`web/secrets/` has zero outbound edges at package-collapse granularity,
yet composer/execution rely on LLM-provider credentials. Either credential
flow happens via `WebSettings` injection at request time (and should be
diagrammed) or via a mechanism the import graph cannot see (which should
be named explicitly). Map to §4.1 SEC2 and synthesis §8.6.

### R10 — **Low** · ARCHITECTURE.md doc-correctness pass · Effort: **S**

Plugin-count drift (KNOW-A35 25 vs A72 46 vs verified 29), 20-vs-21
audit-tables divergence, ADR-007..017 unindexed, schema-mode vocabulary
drift between table and YAML examples, `testing/` misidentification
(KNOW-A18). Each is a small edit; the aggregate is a material onboarding
friction reduction. Map to synthesis §7.13 / §8.2.

### R11 — **Low** · Resolve `errors.py` Tier-1 / Tier-2 split · Effort: **M**

When `contracts/errors.py` next requires material edits, split into
`errors_tier1.py` (raiseable exceptions) and `errors_dtos.py` (frozen
audit DTOs). Don't split-for-the-sake-of-splitting. Map to §3.5 K1.

### R12 — **Low** · Resolve `plugin_context.py:31` TYPE_CHECKING smell · Effort: **S**

Extract `RateLimitRegistryProtocol` into `contracts.config.protocols`,
remove the TYPE_CHECKING block. Cleanly addresses ADR-006d Violation #11.
Map to §3.5 K2 and synthesis §7.11.

---

## §7 What this assessment cannot say

These are explicit gaps in the assessment, not synthesis deferrals.

- **Per-file internal cohesion** of any of the 13 ≥1,500-LOC files. L3
  reads were not performed. The "essential-vs-accidental" question on
  `processor.py`, `config.py`, `dag/graph.py`, `tools.py`, `state.py`,
  `errors.py`, `orchestrator/core.py`, `cli.py`, `execution_repository.py`,
  `azure_batch.py`, `data_flow_repository.py`, `sessions/routes.py`
  remains open.
- **Performance characteristics.** No profiling, no synthetic workload,
  no production-trace analysis. Architecturally hot paths can be inferred
  from the import graph; actually-hot paths cannot.
- **Frontend architecture.** No coverage. Section §4.5 names this as a
  gap, not as a finding.
- **Test-suite architecture.** No coverage. Section §4.4 names this as a
  gap, not as a finding.
- **Operational quality** (CI/CD topology, deployment patterns, observability
  outside Landscape, on-call ergonomics). Out of scope.
- **The workspace's accuracy after merge.** This assessment was performed
  at HEAD = `47d3dd82` (per `git log` at session start). Subsequent
  merges may invalidate any specific file:line citation; the oracle
  re-derivation in §2.2 is the recommended freshness check before relying
  on this document.

---

## §8 Confidence ledger

| Section | Confidence | Why |
|---|---|---|
| §3.1 (engine) | High | Three cluster sources + live LOC verification + clean enforcer |
| §3.2 (core) | High | Cluster source + live verification + KNOW-A* corroboration |
| §3.3 (plugins) | High | Cluster source + oracle edge weights + live LOC |
| §3.4 (web/composer) | High | Cluster source + oracle SCC + live LOC + L1 cross-check |
| §3.5 (contracts) | High | Cluster source + oracle leaf-status confirmation + ADR-006d cited |
| §4.1 (security) | Medium | Defence-in-depth claims are evidence-anchored; SEC2/SEC3 raise gaps the input set cannot close |
| §4.2 (performance) | N/A | Out of scope |
| §4.3 (maintainability) | Medium-High | Mechanical evidence on layer + LOC; "documentation drift" claim corroborated by 5+ KNOW-A* divergences |
| §4.4 (testability) | Low | No test-architecture pass performed |
| §4.5 (frontend gap) | High | The gap itself is verifiable; the consequences are evidence-anchored claims about what cannot be said |
| §5 (under-counts) | High | Live diagnostics directly contradict synthesis §8 file count |
| §6 R1–R12 | Each rec inherits the confidence of its mapped finding | See per-recommendation citations |

---

## §9 Verdict

ELSPETH's architecture is **structurally strong and well-instrumented for
audit-grade integrity**. The mechanical discipline visible in the
codebase — CI-enforced layer model, AST-scanning drift tests, context-manager
state guards, deep-frozen dataclass primitives, allowlist-with-justification
defensive-pattern detection — is genuinely above the median for codebases
of this size and complexity.

It is also a codebase with **named, locatable, addressable structural
debt** in two concentrations: (1) the 7-node `web/*` SCC (R2, High), (2)
the LOC-concentration in 13 files ≥1,500 LOC (R3 / R5, Medium). The debt
is not hidden; the synthesis names most of it, and this assessment adds
the inventory items the synthesis under-counted. One synthesis-flagged
item — the ADR-010 dispatcher verification gap (R1) — was closed during
this assessment by direct L3 read; no remediation required.

The largest unhonest claim available about this codebase would be "the
architecture analysis is complete." It is not. The frontend, the test
architecture, and the per-file cohesion of the 13 large files are
genuinely outside the current workspace's scope. **The honest claim is
that the synthesis-pass and this assessment have produced the strongest
foundation available without those additional passes — and named exactly
where the next passes should land.**

Recommended next-pack sequence:

1. **R8** — STRIDE threat model on the now-stable trust-tier topology
2. **R6** — frontend archaeologist pass (in parallel with R8)
3. **R7** — test-architecture pass (in parallel with R6 / R8)
4. **R5** — L3 deep-dives on `processor.py` + `config.py`
5. **R2** — architecture-pack decision on SCC#4 (after R5 informs)

This sequence prioritises security posture (R8) and the two scope gaps
(R6, R7) before structural refactor decisions (R5, R2). The mechanical
hygiene items (R3, R4, R9, R10, R11, R12) can proceed in parallel
without blocking the strategic sequence. R1 was closed during this
assessment.

---

*End of assessment. Written without diplomatic softening per
architecture-critic discipline. Findings are accurate or wrong; correct
the wrong ones in writing.*
