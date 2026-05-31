# 04 — engine/ cluster report (L2 synthesis)

## Executive synthesis

The `engine/` cluster is the **L2 SDA execution tier** (5 sub-subsystems and 10 standalone modules totalling 17,425 LOC, distributed across 36 files). Its layer position is unambiguous (`enforce_tier_model.py:239` `"engine": 2`); its outbound edges are confined to `{contracts, core}` and verified clean by the L1 oracle. The cluster ships no console scripts of its own — it's invoked through `elspeth.cli` and exposes a 25-name `__all__` from `engine/__init__.py` as the stable contract for L3 callers.

The cluster's defining architectural commitment is the **ADR-010 declaration-trust framework**: a 4-site dispatcher (`pre_emission_check`, `post_emission_check`, `batch_flush_check`, `boundary_check`) drives 7 contract adopters whose mapping to ADRs 007/008/011/012/013/014/016/017 is 1:1 and verified at the file level. A drift-resistant bootstrap manifest (`declaration_contract_bootstrap.py:1-11`) tracked by an AST-scanning unit test (`test_declaration_contract_bootstrap_drift.py`) is the structural defence against silent adopter drift. The dispatcher's "audit-complete" posture (every contract's violations are aggregated before raising, rather than first-fire short-circuiting) is documented inline at `declaration_dispatch.py:1-26`.

The cluster carries the §7 Priority 1 concentration — three files (`orchestrator/core.py` 3,281 LOC, `processor.py` 2,700, `coalesce_executor.py` 1,603) account for ~43% of cluster LOC. The catalog's most consequential finding is that **this concentration is mixed essential-vs-accidental, not uniformly accidental**:

- `orchestrator/core.py` is the *residual* of an in-progress decomposition: the orchestrator's `__init__.py:1-22` docstring records that the previous single 3,000-LOC module has been refactored into six focused siblings (`core.py`, `types.py`, `validation.py`, `export.py`, `aggregation.py`, `outcomes.py`) with stable public API. **Active remediation, not stagnant debt.**
- `coalesce_executor.py` is *essential complexity*: 4 policies × 3 strategies × branch-loss × late-arrivals × checkpoint resume genuinely lives there per the docstring (`coalesce_executor.py:1-5`) and KNOW-A47/A48.
- `processor.py` shows *no L2-visible decomposition* and is the single largest standalone file in the cluster. Its essentiality cannot be answered without an L3 deep-dive — this is the headline open question carried forward from §7.5 Q5.

Layer-import conformance is structurally clean (0 L1 violations, 0 TC warnings inside `engine/`, per the JSON-isolated re-run captured at `temp/layer-conformance-engine.json`). The 69 defensive-pattern findings surfaced by the engine-scoped scanner are pre-allowlisted whole-tree-scoped findings that re-appear under narrowed scope; the catalog routes them into per-entry `Concerns:` with file:line attribution rather than dismissing them. Most fall into legitimate Tier-3 boundary categories (operator-authored expressions in `triggers.py:129/185`, `commencement.py:94`; YAML-load coercion in `dependency_resolver.py:39-51`; protocol discrimination at config dispatch in `dag_navigator.py:219-277`; quarantine-row source-of-unknown-shape in `tokens.py:156`) and are **allowlist-with-justification candidates** rather than bugs. Three concerns warrant closer scrutiny — see "Highest-uncertainty questions" below.

## Layer conformance verdict

**engine/ is fully layer-conformant.** The L1 whole-tree oracle ran clean (`temp/tier-model-oracle.txt`); the engine-scoped JSON-isolated re-run confirms specifically that there are 0 rule-`L1` (upward-import) violations and 0 rule-`TC` (TYPE_CHECKING upward-import) warnings inside `engine/` (`temp/layer-conformance-engine.json`).

A nuance worth recording for future readers: the *raw* engine-scoped run (`temp/layer-check-engine.txt`) exits 1 with 69 "violations". This is **not** a layer-conformance failure — it's the existing allowlist's stale-entry detection firing because allowlist keys are whole-tree-scoped (e.g., `engine/commencement.py:R5:...`) and don't match files when the scan root narrows below the prefix (the keys become `commencement.py:R5:...`). The 69 findings are all defensive-pattern rules (R1/R4/R5/R6/R9), already governed by per-file allowlists in production and re-surfaced only because of the scoping artefact. The whole-tree clean status is authoritative for layer conformance.

## SCC analysis

**N/A.** Engine has no nodes in any Phase 0 SCC (`temp/intra-cluster-edges.json` `stats.sccs_touching_cluster = 0`). The 7-node web/* SCC and the four 2–3-node package-vs-subpackage SCCs (mcp, plugins/transforms/llm, telemetry, tui) all live outside this cluster.

The cluster's intra-engine import structure is, by virtue of being L2, not visible in the L3-only Phase 0 oracle — `temp/intra-cluster-edges.json` is empty by design. Internal coupling within `engine/` is described per sub-subsystem in `02-cluster-catalog.md` "Internal coupling" fields, derived from each file's import section (top ~30 lines) per Δ L2-3.

## The 4 cluster priorities — answers

### 1. KNOW-A70 essential-vs-accidental concentration

**Mixed verdict, file-by-file:**

| File | Essential or accidental | Evidence | L3 follow-up needed? |
|------|-------------------------|----------|---------------------|
| `orchestrator/core.py` | Both — accidental concentration partially remediated; residual is essential | `orchestrator/__init__.py:1-22` decomposition docstring; 6 sibling modules now share what was a single 3,000-LOC module | Yes — quantify what's left in `core.py` after the decomposition vs. what remains as a residual god-class |
| `processor.py` | **Unknown at L2** | No L2-visible decomposition; docstring `:1-9` describes one cohesive responsibility (RowProcessor end-to-end) but the LOC tells a different story | **Yes — this is the priority-1 L3 deep-dive question** |
| `coalesce_executor.py` | Essential | Docstring `:1-5` + KNOW-A47/A48 (4 policies × 3 strategies + branch-loss + late-arrivals + checkpoint) genuinely populates the LOC | No — the cluster catalog's verdict can stand |

The §7.5 amendment did not revise §7 Priority 1's "3–5 hr" effort bracket; this catalog confirms the bracket as appropriate **provided** L3 follow-up scope on `processor.py` and `orchestrator/core.py` is undertaken in a separate L3 pass, not folded into this L2 effort.

### 2. Token identity locus

**Three-locus split (engine-side façade + persistence + call sites):**

- **`engine/tokens.py`** (399 LOC, `TokenManager`): the engine-side **façade** for token lifecycle (create/fork/coalesce/update). The docstring (`tokens.py:1-5`) names this explicitly as "a simplified interface over DataFlowRepository".
- **`core/landscape/data_flow_repository.py`** (out of cluster, 1,590 LOC per L1 deep-dive flag list): the **persistence** of token identity. `tokens.py:19` is the cross-layer import that wires the two together.
- **`engine/processor.py`** + **`engine/orchestrator/core.py`** (both deep-dive flagged): the **call sites** where tokens are minted at fork and consumed at coalesce. Without an L3 deep-dive, the catalog cannot say whether these files mint tokens directly or always go through `TokenManager`.

The L1 catalog and §7 implicitly framed `TokenManager` as the sole authority for token identity; this L2 catalog refines that to "TokenManager is the engine-side façade, not the sole authority." Cross-cluster bookmark: token identity is a shared concern of `engine` and `core`, not engine-only.

### 3. Retry semantics + terminal-state guarantee

**Two-locus split with structural guarantee:**

- **Retry execution loop:** `engine/retry.py` (137 LOC, `RetryManager`). Tenacity-backed; audit-hook contract documented at `retry.py:9-15` ("Each retry attempt must be auditable with the key (run_id, row_id, transform_seq, attempt). The on_retry callback should call recorder.record_retry_attempt()").
- **Retry classification + terminal-state assignment:** `engine/processor.py` (`_execute_transform_with_retry.is_retryable` at `processor.py:1533-1535` is the dispatch site for the R5 isinstance findings against retryable-error types).
- **Structural terminal-state guarantee:** `engine/executors/state_guard.py` (`NodeStateGuard`, **context-manager-as-invariant pattern**). The CLAUDE.md "every row reaches exactly one terminal state" rule is enforced structurally rather than by convention. Locked by `tests/unit/engine/test_state_guard_audit_evidence_discriminator.py` and `tests/unit/engine/test_row_outcome.py`.

The structural-guarantee story is **genuinely good**: a context-manager pattern that makes "exactly one terminal state per token" a structural property rather than a discipline. This is one of the cluster's strongest architectural commitments and would be worth surfacing prominently in any future architecture overview.

### 4. Test-path integrity (CLAUDE.md `from_plugin_instances` rule)

**Scope-conditional finding, qualified divergence from KNOW-C44:**

Spot-checking `tests/unit/engine/`: only `tests/unit/engine/orchestrator/test_phase_error_masking.py` references `ExecutionGraph.from_plugin_instances` or `instantiate_plugins_from_config` in the engine unit-test tree (out of 56 test files). The `tests/unit/engine/conftest.py:23` `MockCoalesceExecutor` carries an explicit "Tests bypass the DAG builder" comment.

CLAUDE.md's rule binds at **integration** scope, not unit (KNOW-C44 reads "integration tests MUST use ExecutionGraph.from_plugin_instances() and instantiate_plugins_from_config()"). `tests/unit/engine/` is unit scope where mocks are tolerated. This is recorded as **`[DIVERGES FROM KNOW-C44]` qualified** rather than a defect — the catalog's framing is "an integration-tier audit is required to determine whether the production-path rule is held there."

There is **no `tests/integration/engine/` directory** (verified via `ls tests/integration/`). Engine integration coverage lives elsewhere in the integration tree and was not re-located within this cluster's scope. The cross-cluster bookmark (below) flags this for synthesis.

## Highest-confidence claims

These three claims are well-evidenced and should propagate verbatim into any post-L2 stitched report.

1. **engine is layer-conformant.** Zero rule-`L1` upward-import violations and zero rule-`TC` TYPE_CHECKING layer warnings inside the cluster (cited from `temp/layer-conformance-engine.json`; consistent with the L1 whole-tree oracle clean status). Outbound dependencies confined to `{contracts, core}`.
2. **The ADR-010 dispatch surface is faithfully implemented and drift-resistant.** 4 dispatch sites × 7 adopters mapped 1:1 to accepted ADRs (007/008/011/012/013/014/016/017); single dispatcher (`declaration_dispatch.py`); closed-set bootstrap manifest (`declaration_contract_bootstrap.py`) tracked by an AST-scanning unit test (`test_declaration_contract_bootstrap_drift.py`); audit-complete posture documented inline at `declaration_dispatch.py:1-26`.
3. **The terminal-state-per-token invariant is structurally guaranteed.** `engine/executors/state_guard.py` (`NodeStateGuard`) implements the "every row reaches exactly one terminal state" rule as a context-manager pattern rather than convention; locked by `test_state_guard_audit_evidence_discriminator.py` and `test_row_outcome.py`.

## Highest-uncertainty questions

These three questions are **the L2-pass agenda for the post-all-L2 synthesis pass**. Each is unanswerable at L2 depth and requires either L3 deep-dive into a flagged file or a cross-cluster audit.

1. **Does `processor.py` (2,700 LOC standalone) have one cohesive responsibility, or multiple accreted concerns?** The docstring claims one (RowProcessor end-to-end); the LOC and the spread of responsibilities visible at the import section (DAG navigation, retry classification, terminal-state assignment, ADR-009b cross-check, batch error handling, quarantine routing) suggests several. **L3 deep-dive into `processor.py` is the priority-1 follow-up.** Without it, KNOW-A70's quality-risk verdict on this file cannot be refined to essential-vs-accidental.
2. **Does `declaration_dispatch.py:137,142` R6 silent-except behaviour honour the inline "every violation is recorded" claim at `declaration_dispatch.py:23-26`?** The catalog's interpretation is that the swallowing is intentional aggregation (collect all violations, then raise), but verifying it requires (a) reading the dispatcher body — currently L3-flagged territory because the surrounding methods are large — and (b) confirming via `test_declaration_dispatch.py` that both `DeclarationContractViolation` and `PluginContractViolation` raised from any registered adopter actually arrive in the dispatcher's aggregation list, not the silent-except branch. **This is test-debt candidate #3 and the highest-stakes verification gap in the cluster.**
3. **Does engine integration testing honour the CLAUDE.md/KNOW-C44 production-path rule?** No `tests/integration/engine/` directory exists, but the integration suite covers engine paths through other directories. The catalog cannot determine from within engine's scope whether those tests use `ExecutionGraph.from_plugin_instances()` consistently or include `MockCoalesceExecutor`-style bypass patterns. **A cross-cluster integration-tier audit is required**; this is not an engine-cluster deliverable.

## Cross-cluster observations for synthesis

(One-line each; the post-all-L2 synthesis pass owns cross-cluster claims, not this cluster.)

- **engine ↔ core.landscape:** `tokens.py:19` imports `DataFlowRepository`. The TokenManager façade pattern means token-identity persistence is fully delegated to core; any L2 core pass should treat `DataFlowRepository` as engine's only persistence partner for token state.
- **engine ↔ contracts.declaration_contracts (ADR-010 payloads):** `executors/transform.py:16-23` imports the ADR-010 declaration-contracts payload typedict surface (`PostEmissionInputs`, `PreEmissionInputs`, `derive_effective_input_fields`). The dispatch payload schema is contracts-defined; engine consumes it. Any L2 contracts pass should expect this surface to be exported.
- **engine ↔ core.expression_parser at three sites:** `triggers.py:24` (TriggerEvaluator condition), `commencement.py:12` (commencement gates), `dependency_resolver.py:14` (depends_on YAML). The expression evaluator is engine-consumed but core-owned; the L2 core pass should locate the implementation in `core/expression_parser.py` (~ which is small per the L1 catalog, post-ADR-006-Phase-1 relocation).
- **engine ↔ contracts.pipeline_runner protocol:** `bootstrap.py` and `dependency_resolver.py` consume `PipelineRunner` (contracts-defined, engine-implemented at orchestration scope). The runner protocol is the contract that lets `bootstrap.resolve_preflight()` be the shared CLI/programmatic entry.
- **engine integration testing — no in-cluster directory.** No `tests/integration/engine/` exists. Integration coverage of engine paths must live somewhere in the integration tree (likely under cluster-specific integration directories such as `tests/integration/<plugins-area>/`). A cross-cluster integration-tier audit is required to verify the KNOW-C44 production-path rule.

## L1 cross-references

This report supplements:

- `02-l1-subsystem-map.md §3 (engine/)` — ratifies and refines the L1 entry's "Highest-risk concern" (the 3-file 1,500+ LOC concentration), specifically nuancing KNOW-A70 to a mixed essential-vs-accidental verdict.
- `04-l1-summary.md §7 Priority 1` — confirms the priority-1 ranking is appropriate; the 3–5 hr effort bracket holds **provided** L3 deep-dive on `processor.py` and `orchestrator/core.py` is scheduled separately.
- `04-l1-summary.md §7.5 Q5` — partially answers ("processor.py: unknown at L2 depth; orchestrator/core.py: residual of in-progress decomposition; coalesce_executor.py: essential"), with the residual-vs-essential split for processor.py carried forward as the headline L3 question.
- `04-l1-summary.md §7.5 standing note` — engine is L2, so the F5 unconditional-runtime-coupling note doesn't apply within the cluster (it applies to engine's L3 callers in their own clusters); the layer-check artefact's 0 TC warnings confirms engine itself has no TYPE_CHECKING-guarded coupling at the L0/L1 boundary either.
