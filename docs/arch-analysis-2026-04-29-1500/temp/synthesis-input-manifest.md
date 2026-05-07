# Synthesis Input Manifest — Phase 8

This is the exhaustive set of cluster-level claims the Phase 8 stitched report may draw on. Cluster claims are reproduced verbatim with their source-section citations. Cross-cluster synthesis claims (the output of Phase 8) MUST cite at least one entry from this manifest plus either a second cluster's manifest entry, the Phase 0 L3 oracle, or the L1 dispatch queue (§7/§7.5 of 04-l1-summary.md).

## Provenance per cluster

For each cluster, the table below names where in the 04-cluster-report.md the three section types live (heading text + actual § number observed).

| Cluster | Confidence § | Uncertainty § | Cross-cluster § |
|---------|--------------|----------------|-----------------|
| engine | (unnumbered) "Highest-confidence claims" | (unnumbered) "Highest-uncertainty questions" | (unnumbered) "Cross-cluster observations for synthesis" |
| core | §3 "Highest-confidence claims (top 3 — for stitched-report propagation)" | §4 "Highest-uncertainty questions (top 3 — agenda for post-L2 synthesis)" | §6 "Cross-cluster observations for synthesis (Δ L2-4 deferral channel)" |
| composer | §5 "Highest-confidence claims (top 3 — for stitched report)" | §6 "Highest-uncertainty questions (top 3 — agenda for post-L2 synthesis)" | §7 "Cross-cluster observations for synthesis (deferred from Δ L2-4)" |
| plugins | §10 "Highest-confidence claims" | §11 "Highest-uncertainty questions" | §9/§12 "Cross-cluster observations for synthesis" |
| contracts | (unnumbered) "Highest-confidence claims" | (unnumbered) "Highest-uncertainty questions" | (unnumbered) "Cross-cluster observations for synthesis" |

## Entries

### engine cluster

[engine] [confidence] **engine is layer-conformant.** Zero rule-`L1` upward-import violations and zero rule-`TC` TYPE_CHECKING layer warnings inside the cluster (cited from `temp/layer-conformance-engine.json`; consistent with the L1 whole-tree oracle clean status). Outbound dependencies confined to `{contracts, core}`.
  Source: clusters/engine/04-cluster-report.md "Highest-confidence claims" (item 1)

[engine] [confidence] **The ADR-010 dispatch surface is faithfully implemented and drift-resistant.** 4 dispatch sites × 7 adopters mapped 1:1 to accepted ADRs (007/008/011/012/013/014/016/017); single dispatcher (`declaration_dispatch.py`); closed-set bootstrap manifest (`declaration_contract_bootstrap.py`) tracked by an AST-scanning unit test (`test_declaration_contract_bootstrap_drift.py`); audit-complete posture documented inline at `declaration_dispatch.py:1-26`.
  Source: clusters/engine/04-cluster-report.md "Highest-confidence claims" (item 2)

[engine] [confidence] **The terminal-state-per-token invariant is structurally guaranteed.** `engine/executors/state_guard.py` (`NodeStateGuard`) implements the "every row reaches exactly one terminal state" rule as a context-manager pattern rather than convention; locked by `test_state_guard_audit_evidence_discriminator.py` and `test_row_outcome.py`.
  Source: clusters/engine/04-cluster-report.md "Highest-confidence claims" (item 3)

[engine] [uncertainty] **Does `processor.py` (2,700 LOC standalone) have one cohesive responsibility, or multiple accreted concerns?** The docstring claims one (RowProcessor end-to-end); the LOC and the spread of responsibilities visible at the import section (DAG navigation, retry classification, terminal-state assignment, ADR-009b cross-check, batch error handling, quarantine routing) suggests several. **L3 deep-dive into `processor.py` is the priority-1 follow-up.** Without it, KNOW-A70's quality-risk verdict on this file cannot be refined to essential-vs-accidental.
  Source: clusters/engine/04-cluster-report.md "Highest-uncertainty questions" (item 1)

[engine] [uncertainty] **Does `declaration_dispatch.py:137,142` R6 silent-except behaviour honour the inline "every violation is recorded" claim at `declaration_dispatch.py:23-26`?** The catalog's interpretation is that the swallowing is intentional aggregation (collect all violations, then raise), but verifying it requires (a) reading the dispatcher body — currently L3-flagged territory because the surrounding methods are large — and (b) confirming via `test_declaration_dispatch.py` that both `DeclarationContractViolation` and `PluginContractViolation` raised from any registered adopter actually arrive in the dispatcher's aggregation list, not the silent-except branch. **This is test-debt candidate #3 and the highest-stakes verification gap in the cluster.**
  Source: clusters/engine/04-cluster-report.md "Highest-uncertainty questions" (item 2)

[engine] [uncertainty] **Does engine integration testing honour the CLAUDE.md/KNOW-C44 production-path rule?** No `tests/integration/engine/` directory exists, but the integration suite covers engine paths through other directories. The catalog cannot determine from within engine's scope whether those tests use `ExecutionGraph.from_plugin_instances()` consistently or include `MockCoalesceExecutor`-style bypass patterns. **A cross-cluster integration-tier audit is required**; this is not an engine-cluster deliverable.
  Source: clusters/engine/04-cluster-report.md "Highest-uncertainty questions" (item 3)

[engine] [cross-cluster] **engine ↔ core.landscape:** `tokens.py:19` imports `DataFlowRepository`. The TokenManager façade pattern means token-identity persistence is fully delegated to core; any L2 core pass should treat `DataFlowRepository` as engine's only persistence partner for token state.
  Source: clusters/engine/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 1)

[engine] [cross-cluster] **engine ↔ contracts.declaration_contracts (ADR-010 payloads):** `executors/transform.py:16-23` imports the ADR-010 declaration-contracts payload typedict surface (`PostEmissionInputs`, `PreEmissionInputs`, `derive_effective_input_fields`). The dispatch payload schema is contracts-defined; engine consumes it. Any L2 contracts pass should expect this surface to be exported.
  Source: clusters/engine/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 2)

[engine] [cross-cluster] **engine ↔ core.expression_parser at three sites:** `triggers.py:24` (TriggerEvaluator condition), `commencement.py:12` (commencement gates), `dependency_resolver.py:14` (depends_on YAML). The expression evaluator is engine-consumed but core-owned; the L2 core pass should locate the implementation in `core/expression_parser.py` (~ which is small per the L1 catalog, post-ADR-006-Phase-1 relocation).
  Source: clusters/engine/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 3)

[engine] [cross-cluster] **engine ↔ contracts.pipeline_runner protocol:** `bootstrap.py` and `dependency_resolver.py` consume `PipelineRunner` (contracts-defined, engine-implemented at orchestration scope). The runner protocol is the contract that lets `bootstrap.resolve_preflight()` be the shared CLI/programmatic entry.
  Source: clusters/engine/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 4)

[engine] [cross-cluster] **engine integration testing — no in-cluster directory.** No `tests/integration/engine/` exists. Integration coverage of engine paths must live somewhere in the integration tree (likely under cluster-specific integration directories such as `tests/integration/<plugins-area>/`). A cross-cluster integration-tier audit is required to verify the KNOW-C44 production-path rule.
  Source: clusters/engine/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 5)

### core cluster

[core] [confidence] **`core/` is fully layer-conformant; the 4-layer model is mechanically respected at L1.** Verified by `enforce_tier_model.py check --root src/elspeth/core --allowlist <production>` (205 findings, all R5/R6/R1/R4 defensive-pattern items already governed by per-file allowlists; **0 L1, 0 TC**), corroborated by re-run with empty allowlist (identical histogram). Public surface of `core/__init__.py` (100 LOC) confirms outbound to `contracts/` only — `IntegrityError`, `PayloadStore`, `ResumeCheck`, `ResumePoint`, plus DTOs are re-exported from `contracts/`. **Implication:** Any future cross-layer need MUST follow the `KNOW-ADR-006d` "Violation #11 Protocol" (move down → extract primitive → restructure caller → never lazy-import). The single observed lazy intra-cluster import (`dependency_config.py:63` → `ExpressionParser`) is intra-`core/` and not a layer violation.
  Source: clusters/core/04-cluster-report.md §3 (item 1)

[core] [confidence] **The Landscape sub-area is the cluster's hub and matches the documented 4-repository facade pattern.** `landscape/__init__.py` (154 LOC, read in full) re-exports `RecorderFactory` (the facade per `KNOW-A29`), `LandscapeDB`, and exactly the 4 repositories named in `KNOW-A30` (`DataFlowRepository`, `ExecutionRepository`, `QueryRepository`, `RunLifecycleRepository`). `landscape/schema.py` defines **20 tables** (verified by grep: runs, nodes, edges, rows, tokens, token_outcomes, token_parents, node_states, operations, calls, artifacts, routing_events, batches, batch_members, batch_outputs, validation_errors, transform_errors, checkpoints, secret_resolutions, preflight_results) — `[DIVERGES FROM KNOW-A24]` which claims 21. Repositories are **NOT** re-exported through the cluster's `core/__init__.py`; callers reach them via `elspeth.core.landscape.*` per the encapsulation discipline. **Implication:** the documented "facade + 4 repositories" architecture is real and load-bearing — refactoring proposals that bypass `RecorderFactory` should be challenged.
  Source: clusters/core/04-cluster-report.md §3 (item 2)

[core] [confidence] **A "Protocol-based no-op parity" pattern recurs across `core/` and is a deliberate offensive-programming choice.** `EventBus`/`NullEventBus` (`events.py:14–28, 88–111`) and `RateLimiter`/`NoOpLimiter` (`rate_limit/registry.py:33–66`) both implement a protocol structurally rather than via inheritance. The `NullEventBus` docstring is explicit (`events.py:88–103`): "If someone subscribes expecting callbacks, inheritance would hide the bug. Protocol-based design makes the no-op behavior explicit." Combined with the pervasive use of `freeze_fields` / `__post_init__` validation per `KNOW-C61`–`KNOW-C65` and the offensive-programming `__post_init__` checks in `dag/models.py:151–193` and `dependency_config.py:144–155`, this is a coherent design discipline. **Implication:** new no-op or null-object additions to `core/` MUST follow the Protocol pattern; inheritance from the real class is the wrong default.
  Source: clusters/core/04-cluster-report.md §3 (item 3)

[core] [uncertainty] **`config.py` (2,227 LOC) cohesion — essential or accidental complexity?** The single Pydantic settings file holds 12+ child dataclasses (`CheckpointSettings`, `ConcurrencySettings`, `DatabaseSettings`, `ElspethSettings`, `LandscapeExportSettings`, `LandscapeSettings`, `PayloadStoreSettings`, `RateLimitSettings`, `RetrySettings`, `SecretsConfig`, `ServiceRateLimit`, `SinkSettings`, `SourceSettings`, `TransformSettings`) plus the `load_settings()` loader. Pydantic settings tend to concentrate for cross-validation reasons, but 2,227 LOC of single-file configuration is substantial. **Open question:** does internal structure factor cleanly (e.g., per-domain validator clusters, source/transform/sink groupings) or has it accreted by addition? **Resolution path:** L3 deep-dive on `config.py` followed by an architecture-pack proposal (split or keep) — **not** a `core/` archaeology decision.
  Source: clusters/core/04-cluster-report.md §4 (item 1)

[core] [uncertainty] **Contracts/core boundary post-ADR-006 (L1 open question Q1).** This pass observed `core/`'s outbound imports from `contracts/` and found them concentrated in expected primitive surfaces (`payload_store`, `errors`, `freeze`, `hashing`, `schema`, `schema_contract`, `secrets`, `security`). The structural side (no upward imports, layer-clean) is verified. The semantic side — *should* the responsibility cut be different? are there primitives currently in `core/` that belong in `contracts/`, or vice-versa? — requires reading both clusters and is a post-L2 synthesis concern. **Specific candidate for review:** `core/secrets.py` (124 LOC, runtime resolver) lives at `core/` root while `core/security/{secret_loader,config_secrets}.py` (529 LOC combined) live in the subpackage. Their topical relationship (both about secrets) but role disjointness (runtime vs config-time) is intentional but may benefit from explicit naming or co-location.
  Source: clusters/core/04-cluster-report.md §4 (item 2)

[core] [uncertainty] **`dag/graph.py` (1,968 LOC) cascade-prone risk — concrete blast radius?** The §7 P3 "cascade-prone" framing is qualitatively correct: `ExecutionGraph` is consumed by every executor in `engine/`, by `web/composer/_semantic_validator.py` (per L3 oracle), by `web/execution/validation.py`, by `core/checkpoint/{manager,compatibility}`, and indirectly by every plugin via the schema-contract validation flow. **Open question:** what is the test surface that locks `ExecutionGraph`'s public contract? `tests/unit/core/dag/test_graph.py` and `test_graph_validation.py` exist; their assertion density vs the file's behavioural surface area is the deferred deep-dive question. The `tests/unit/core/dag/test_models_post_init.py` evidence is encouraging (it locks the `NodeInfo` invariants), but `graph.py` itself was not opened by this pass.
  Source: clusters/core/04-cluster-report.md §4 (item 3)

[core] [cross-cluster] **(Synthesis-1) Contracts↔core boundary inventory.** `core/` imports the following identifiers from `contracts/`, observed during this pass: `IntegrityError`, `PayloadStore` (Protocol), `PayloadNotFoundError`, `AuditIntegrityError`, `TIER_1_ERRORS`, `BatchPendingError`, `Operation`, `Artifact`, `Batch`, `BatchMember`, `BatchOutput`, `Call`, `CallStatus`, `CallType`, `Checkpoint`, `ContractAuditRecord`, `Edge`, `FieldAuditRecord`, `Node`, `NodeState`, `NodeStateCompleted`, `NodeStateFailed`, `NodeStateOpen`, `NodeStateStatus`, `ReproducibilityGrade`, `ResumeCheck`, `ResumePoint`, `RoutingEvent`, `RoutingSpec`, `Row`, `RowLineage`, `Run`, `RunStatus`, `Token`, `TokenParent`, `ValidationErrorWithContract`, `SecretResolutionInput`, `CANONICAL_VERSION` (hashing), `deep_freeze`, `deep_thaw`, `freeze_fields`, `require_int` (freeze), `FieldDefinition`, `SchemaConfig` (schema), `PipelineRow`, `SchemaContract` (schema_contract), `ResolvedSecret`, `WebSecretResolver` (secrets), `secret_fingerprint`, `get_fingerprint_key` (security), `NodeType` (enums), `CoalesceName`, `NodeID` (types), and the `aggregation_checkpoint` / `coalesce_checkpoint` typed-dict modules. The contracts cluster's pass should verify that **every name above is present in its `__init__.py` `__all__` list** — if any is missing, that's a contracts-side debt item.
  Source: clusters/core/04-cluster-report.md §6 (Synthesis-1)

[core] [cross-cluster] **(Synthesis-2) `KNOW-A24` 20-vs-21-tables question.** `core/landscape/schema.py` defines 20 tables; the doc says 21. Possible explanations: (a) one table was renamed/dropped after `KNOW-A24` was written; (b) one is conditionally created (e.g., dialect-specific); (c) the doc was always off by one. **Defer to doc-correctness pass.** The contracts cluster does not own this question (table definitions live in `core/landscape/schema.py`, not `contracts/`).
  Source: clusters/core/04-cluster-report.md §6 (Synthesis-2)

[core] [cross-cluster] **(Synthesis-3) Engine cluster will need the `core/` outbound surface this pass enumerates.** When the engine cluster catalog asserts "engine imports `RecorderFactory`, `ExecutionGraph`, `compute_full_topology_hash`, `CheckpointManager`, …" it should cite this report's section 6 for the `contracts/`-side primitives that engine reaches **through** core (e.g. engine → core → contracts.errors.TIER_1_ERRORS for the `tier_1_error` registry per `KNOW-ADR-010b`).
  Source: clusters/core/04-cluster-report.md §6 (Synthesis-3)

[core] [cross-cluster] **(Synthesis-4) `core/secrets.py` ↔ `web/composer/`** — the runtime secret-ref resolver in `core/secrets.py` is consumed by the web composer when threading `{"secret_ref": ...}` references through resolved configs. The composer cluster's catalog should record `web/composer/* → core/secrets` as one of its outbound `core/` edges.
  Source: clusters/core/04-cluster-report.md §6 (Synthesis-4)

[core] [cross-cluster] **(Synthesis-5) MCP/composer_mcp cluster's separation rationale is reinforced by this pass.** `core/landscape/__init__.py` is the read-only audit DB surface (`elspeth-mcp` consumes this per `KNOW-C35`), distinct from `composer_mcp/` which has no Landscape coupling at all. This pass does not assert anything about the MCP clusters — merely confirms that the structural separation in `core/` (Landscape sub-area is encapsulated, `RecorderFactory`-fronted) supports the L1 "do not merge" guidance.
  Source: clusters/core/04-cluster-report.md §6 (Synthesis-5)

### composer cluster

[composer] [confidence] **The composer cluster has one composer state machine and three internal consumers.** `web/composer/state.py` (1,710 LOC) and `tools.py` (3,804 LOC) own the pipeline-composition state and tool surface. This state is consumed by `composer_mcp/` (transport, weight 12), `web/sessions/` (persistence, weight 15), and `web/execution/` (validation, weight 9). F1 stands at the symbol level (`composer_mcp/server.py:1–40` imports the state types directly); F1's "thin transport" framing is correct but understates the structural role — `web/composer/` is the cluster's data backbone, not just an MCP target.
  Source: clusters/composer/04-cluster-report.md §5 (item 1)

[composer] [confidence] **The 7-node SCC is the FastAPI app-factory pattern made structural.** `web/app.py:create_app(...)` outwardly imports every sub-package's router (the wiring leg); sub-packages reach back via `from elspeth.web.config import WebSettings` and `run_sync_in_worker` (the shared-infrastructure leg). Both directions are intentional. The cycle is load-bearing in its current form, and decomposition is a non-trivial refactoring decision left to the architecture-pack pass.
  Source: clusters/composer/04-cluster-report.md §5 (item 2)

[composer] [confidence] **The cluster has 0 inbound edges from any other cluster.** Confirmed by `temp/intra-cluster-edges.json:cross_cluster_inbound_edges = []`. The composer cluster is consumed only by its two console-script entry points (`elspeth-web`, `elspeth-composer`), not by library code elsewhere. This is the structural signature of an application-surface cluster, and it tightens what the post-L2 synthesis pass can plausibly say about cluster-level coupling: there are no surprise back-references to defer.
  Source: clusters/composer/04-cluster-report.md §5 (item 3)

[composer] [uncertainty] **What is the `web/execution → .` (root `elspeth` package) edge importing?** Weight 3, sample sites `web/execution/service.py:30,805` and `validation.py:24`. The catalog cannot diagnose this without file inspection at those lines. If it's a re-export of a public symbol from `elspeth/__init__.py`, that's benign. If it's something else (e.g., a deferred-import hack to bypass an explicit cluster dependency), that's a different finding.
  Source: clusters/composer/04-cluster-report.md §6 (item 1)

[composer] [uncertainty] **How does the composer cluster's secrets surface compose with cross-cluster secret handling?** `web/secrets/` has zero outbound edges to other clusters at the package-collapse granularity, yet composer/execution rely on LLM-provider credentials that are presumably loaded through the secrets surface and then handed to plugin code. Is the credential flow happening via `WebSettings` injection at request time (not visible to the import graph)? L3 inspection territory.
  Source: clusters/composer/04-cluster-report.md §6 (item 2)

[composer] [uncertainty] **Why is `web/sessions → web/composer` weight 15 (joint-heaviest intra-cluster edge)?** The catalog interprets this as "sessions persists composer drafts" based on the file names (`engine.py`, `service.py:_assert_state_in_session`), but the symbol-level evidence has not been inspected. Confirming the data-flow direction (sessions reads/writes composer state types vs composer reads session metadata) is L3 scope.
  Source: clusters/composer/04-cluster-report.md §6 (item 3)

[composer] [cross-cluster] **The cluster does not import directly from `engine/` at the package-collapse granularity.** It routes through `plugins/infrastructure/` (3 of the 6 outbound edges target this) and through `plugins/infrastructure/`-routed plugin metadata. This is consistent with the L1 catalog's "engine instantiates plugins via the registry" claim, but the synthesis pass should confirm the same pattern holds in the engine and plugins clusters' L2 catalogs.
  Source: clusters/composer/04-cluster-report.md §7 (bullet 1)

[composer] [cross-cluster] **The conditional outbound edge `web/execution → telemetry` (weight 1) is the cluster's only conditional cross-cluster dependency.** Whether telemetry-conditional imports are a cluster-specific pattern or a project-wide pattern is a synthesis question; this catalog records yes for this cluster only.
  Source: clusters/composer/04-cluster-report.md §7 (bullet 2)

[composer] [cross-cluster] **`mcp/` and `composer_mcp/` are confirmed independent siblings at the import level (F2).** This catalog reaffirms the L1 "do not merge" guidance for `mcp/` but does not analyse `mcp/`. The synthesis pass should validate that `mcp/`'s own L2 pass (if undertaken) does not contradict this.
  Source: clusters/composer/04-cluster-report.md §7 (bullet 3)

[composer] [cross-cluster] **The cluster's two console-script entry points are the only inbound consumers.** No other cluster imports anything from `web/` or `composer_mcp/`. The synthesis pass can therefore treat the composer cluster as a *terminal* cluster in the import-direction sense — it consumes upstream clusters but is not consumed by them.
  Source: clusters/composer/04-cluster-report.md §7 (bullet 4)

### plugins cluster

[plugins] [confidence] **plugins/ is layer-conformant and structurally clean.** Whole-tree `enforce_tier_model.py check` runs clean; intra-cluster edges (23) all flow toward `infrastructure/` (the spine); 0 outbound L3↔L3 edges; F3 reading-order verified empirically. **Confidence: High** — oracle-cited at every step, byte-equivalent on re-derivation.
  Source: clusters/plugins/04-cluster-report.md §10 (item 1)

[plugins] [confidence] **Trust-tier discipline is documented, repeated, and structurally encoded.** Every source module repeats the "ONLY place coercion is allowed" contract; every sink module repeats the "wrong types = upstream bug = crash" contract; the discipline is also encoded in the `allow_coercion` config flag. The contract is enforced at both layers. **Confidence: High** — verbatim docstring matches across plugin files, citable file:line.
  Source: clusters/plugins/04-cluster-report.md §10 (item 2)

[plugins] [confidence] **SCC #1 is module-level only and structurally minimal.** Provider-registry pattern with deferred runtime instantiation; both sides need each other for type sharing; the only decomposition options touch `infrastructure/` or introduce indirection. **Confidence: High** — import sites enumerated by file:line; runtime decoupling cited from `transform.py:9-13`.
  Source: clusters/plugins/04-cluster-report.md §10 (item 3)

[plugins] [uncertainty] **Is the SCC #1 cycle worth breaking, given that runtime coupling is already deferred?** The cycle is import-time only; the architecture pack will need to compare the cost of moving shared types into `infrastructure/` (further bloating an already-composite spine) versus the cost of leaving the cycle visible.
  Source: clusters/plugins/04-cluster-report.md §11 (item 1)

[plugins] [uncertainty] **Does the documented trust-tier discipline hold at runtime under all execution paths?** Verbal/structural enforcement is in place; cross-cluster invariant tests are not. A targeted runtime probe (e.g., a test fixture that injects a transform observed to coerce and asserts the run fails) would close the gap. This is in the test-debt list but its priority is uncertain.
  Source: clusters/plugins/04-cluster-report.md §11 (item 2)

[plugins] [uncertainty] **Is the 29-vs-25 plugin count a doc-rot artefact or a signal of governance drift?** Four post-doc plugins were added without a doc update. The architecture-pack pass should decide whether this is acceptable churn or whether plugin-count is a controlled invariant. KNOW-A72's "46" remains unexplained.
  Source: clusters/plugins/04-cluster-report.md §11 (item 3)

[plugins] [cross-cluster] **`web/composer → plugins/infrastructure (w=22)`** is the heaviest cross-cluster inbound edge to plugins/. Synthesis owns: what does composer need from infrastructure that warrants a single edge of this weight?
  Source: clusters/plugins/04-cluster-report.md §9 (bullet 1) / §12 (bullet 1)

[plugins] [cross-cluster] **`. (cli root) → plugins/infrastructure (w=7)` and `. → plugins/sources (w=2)`** confirm KNOW-P22 (cli registry pattern). Synthesis owns: whether the cli's import surface to plugins/ is a coupling worth re-reviewing post-ADR-006.
  Source: clusters/plugins/04-cluster-report.md §9 (bullet 2) / §12 (bullet 2)

[plugins] [cross-cluster] **`testing → plugins/infrastructure (w=4)`** suggests the testing harness has hard imports into plugin spine. Synthesis owns: whether the harness should depend on protocols (`contracts/`) instead.
  Source: clusters/plugins/04-cluster-report.md §9 (bullet 3) / §12 (bullet 3)

[plugins] [cross-cluster] **SCC #1 vs other L3 SCCs.** The plugin SCC #1 is one of five L3 SCCs (mcp/, plugins/transforms/llm/, telemetry/, tui/, web/). Synthesis owns: do they share a common cause (import-time registry pattern) or are they incidental?
  Source: clusters/plugins/04-cluster-report.md §9 (bullet 4) / §12 (bullet 4)

[plugins] [cross-cluster] **R-rule findings density at the L3 boundary.** plugins/ has 291 cluster-scoped R-rule findings (R5=140, R6=52). Synthesis owns: cluster-by-cluster comparison of R-rule density may identify boundary-handling discipline gradients.
  Source: clusters/plugins/04-cluster-report.md §9 (bullet 5) / §12 (bullet 5)

### contracts cluster

[contracts] [confidence] **The L0 leaf invariant is mechanically confirmed.** `enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` exits 0 with "No bug-hiding patterns detected"; `temp/intra-cluster-edges.json` shows zero outbound edges; `temp/layer-conformance-contracts.json` records `layer_import_violations_L1: []` and `type_checking_layer_warnings_TC: []`. KNOW-A53 stands. The single TYPE_CHECKING annotation to `core/` (`plugin_context.py:31`) is permitted and annotation-only.
  Source: clusters/contracts/04-cluster-report.md "Highest-confidence claims" (item 1)

[contracts] [confidence] **ADR-006 phase artefacts are visible inside the cluster and the post-relocation boundary is materially clean.** Phase 2's `hashing.py` extraction (`:8-11`), Phase 4's `RuntimeServiceRateLimit` dataclass (`config/runtime.py:291`), and Phase 5's CI gate (`enforce_tier_model.py:237`) are all present. The `__init__.py:79-87` and `:369-370` comment blocks encode the post-ADR-006 boundary as institutional memory. KNOW-ADR-006a–d ratified.
  Source: clusters/contracts/04-cluster-report.md "Highest-confidence claims" (item 2)

[contracts] [confidence] **The ADR-010 declaration-trust framework's L0 surface is complete and the L0/L2 split is clean.** The contracts cluster defines the vocabulary (`AuditEvidenceBase` ABC, `@tier_1_error` decorator + frozen registry, `DeclarationContract` 4-site framework with bundle types and payload-schema H5, secret-scrub last-line-of-defence); the engine cluster's already-mapped 4-site dispatcher × 7 adopters consumes this vocabulary end-to-end. Two engine-cluster cross-cluster bookmarks are closed by this pass.
  Source: clusters/contracts/04-cluster-report.md "Highest-confidence claims" (item 3)

[contracts] [uncertainty] **Does the `plugin_context.py:31` TYPE_CHECKING import to `core.rate_limit.RateLimitRegistry` warrant an "extract primitive" resolution per ADR-006d's Violation #11 Protocol?** The annotation is structurally permitted but it's the only cross-layer reference in the cluster; an extracted `RateLimitRegistryProtocol` in `contracts.config.protocols` would eliminate the TYPE_CHECKING block and tighten the L0/L1 boundary. **Owner: architecture pack.** This is the strongest Q1 evidence the catalog surfaces.
  Source: clusters/contracts/04-cluster-report.md "Highest-uncertainty questions" (item 1)

[contracts] [uncertainty] **Should `errors.py` (1,566 LOC) be split, and if so, along which seam?** The file holds Tier-1 raiseable exceptions, Tier-2 frozen audit DTOs, structured-reason TypedDicts, and re-exported `FrameworkBugError`. The Tier-1/Tier-2 distinction is currently encoded by inline comments (`errors.py:34` `# TIER-2: Frozen audit DTO ...`); a CI-enforced split (e.g., `contracts/errors_tier1.py` vs `contracts/errors_dtos.py`) would mechanise the discipline. **Owner: architecture pack.** This is also the highest-priority L3 deep-dive candidate in the cluster.
  Source: clusters/contracts/04-cluster-report.md "Highest-uncertainty questions" (item 2)

[contracts] [uncertainty] **Should the schema-contract subsystem (Catalog Entry 8, 8 files / ~3,500 LOC) be promoted from "top-level files" to a `contracts/schema_contracts/` sub-package?** The internal cohesion is high (all 8 files reference `FieldContract` / `SchemaContract` / `PipelineRow`); the names don't make their layering self-evident; a sub-package would mirror the `config/` partition and clarify the cluster's internal structure. The L1 entry summarised the schema-contract surface as one item; at L2 depth it is clearly a sub-cluster. **Owner: architecture pack.** Not blocking; pure organisational hygiene.
  Source: clusters/contracts/04-cluster-report.md "Highest-uncertainty questions" (item 3)

[contracts] [cross-cluster] **contracts ↔ core (TYPE_CHECKING smell at `plugin_context.py:31`):** the only cross-layer reference in the cluster; candidate for ADR-006d "Violation #11" remediation. Owner: synthesis + architecture pack.
  Source: clusters/contracts/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 1)

[contracts] [cross-cluster] **contracts ↔ engine (ADR-010 dispatch surface):** L0 vocabulary is complete; engine-cluster catalog entry 2 enumerated the 4-site × 7-adopter mapping; the `pipeline_runner` Protocol bookmark is also closed. Owner: synthesis (already aligned).
  Source: clusters/contracts/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 2)

[contracts] [cross-cluster] **contracts ↔ core/landscape (audit DTO surface):** `audit.py` (922 LOC, header-only at this pass) is the L0-side of the Landscape audit-trail row contract; the L1 core cluster pass owns the L1-side write/read mechanics. Owner: synthesis.
  Source: clusters/contracts/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 3)

[contracts] [cross-cluster] **contracts ↔ core/checkpoint (checkpoint family):** four checkpoint-family dataclasses (Catalog Entry 12) are L0 DTOs persisted by the L1 checkpoint repository. Owner: synthesis.
  Source: clusters/contracts/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 4)

[contracts] [cross-cluster] **`type_normalization.py` R5 findings — trust-boundary correctness:** 184 isinstance findings on this single file are at the runtime-type-normalization trust boundary; whole-tree allowlist accepts them. The synthesis pass should note that this file is the cluster's densest correctness surface and a likely candidate for any future R-rule policy review. Owner: synthesis.
  Source: clusters/contracts/04-cluster-report.md "Cross-cluster observations for synthesis" (bullet 5)

## Summary counts

- engine:    3 confidence + 3 uncertainty + 5 cross-cluster = 11 entries
- core:      3 confidence + 3 uncertainty + 5 cross-cluster = 11 entries
- composer:  3 confidence + 3 uncertainty + 4 cross-cluster = 10 entries
- plugins:   3 confidence + 3 uncertainty + 5 cross-cluster = 11 entries
- contracts: 3 confidence + 3 uncertainty + 5 cross-cluster = 11 entries
- TOTAL:     15 confidence + 15 uncertainty + 24 cross-cluster = 54 entries

## Notes

- Verbatim text means the cluster report's actual prose. If a claim has sub-bullets or evidence, include them as a sub-blockquote under the entry. Do not paraphrase; do not summarise.
- For multi-sentence claims, keep all sentences. For long paragraphs (>200 words), keep the first 2 sentences and add "[...truncated; full text in source §<N>]" — but this should be rare; cluster claims are typically pithy.
- If a cluster's "cross-cluster observations" section is empty or absent, write "[<cluster>] [cross-cluster] (none — section absent or empty)" with the source reference still cited so the next consumer can verify.
- The plugins cluster's "Cross-cluster observations" appears twice in its report (§9 and §12 with §12 explicitly noted as "Repeated from §9 for the validator"); both citations are retained in the source line for each entry.
- The contracts cluster's report duplicates the three "Highest-confidence claims" / "Highest-uncertainty questions" headings (once in the body of the cross-cluster section, once at the explicit named subsection near end-of-file). The verbatim text used here is from the explicit named subsections at end-of-file (lines 132–161), which match the body content.
