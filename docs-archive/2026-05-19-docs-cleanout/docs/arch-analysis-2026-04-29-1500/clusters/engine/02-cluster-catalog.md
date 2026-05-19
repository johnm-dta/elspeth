# 02 — engine/ cluster catalog (L2 sub-subsystem map)

## Conventions

- **Layer:** `engine/` is **L2** [`scripts/cicd/enforce_tier_model.py:239` `"engine": 2`]. Outbound subset permitted to `{contracts, core}` only; downward-flowing imports verified clean by the L1 oracle (`temp/tier-model-oracle.txt`).
- **Intra-cluster oracle:** `[ORACLE: temp/intra-cluster-edges.json stats.intra_edge_count = 0, intra_node_count = 0]`. Engine is L2; the cluster oracle is derived from `temp/l3-import-graph.json`, which filters L3-only nodes — emptiness is **expected** and constitutes evidence that engine has no L3 graph footprint, NOT evidence of internal decoupling. L2 internal coupling is derived per entry from each file's import section per Δ L2-3.
- **Defensive-pattern scanner artefact:** `[ORACLE: temp/layer-check-engine-empty-allowlist.txt — 69 findings under R1 (dict.get), R4 (broad-except), R5 (isinstance), R6 (silent-except), R9 (dict.pop-default)]`. **This is `enforce_tier_model.py` running in defensive-pattern mode with an empty allowlist (CLAUDE.md sense), NOT the layer-import enforcer (ADR-006 Phase 5 sense).** Layer-import conformance is inherited from the L1 oracle, which ran clean. The 69 findings drive per-entry `Concerns:` content below; allowlist mechanism (`config/cicd/enforce_tier_model/`) governs production status of each finding.
- **L3-deep-dive flags (Δ L2-3):** `processor.py` (2,700 LOC), `coalesce_executor.py` (1,603 LOC), and `orchestrator/core.py` (3,281 LOC, inside the orchestrator sub-package) are named-and-deferred. Their entries derive responsibility from prior docs + first-30-lines docstring/imports only.
- **Cluster-internal SCC handling:** N/A — engine has no nodes in any Phase 0 SCC (`stats.sccs_touching_cluster = 0`).
- **Test-path integrity probe (cluster priority 4):** Spot-check of `tests/unit/engine/`: only `tests/unit/engine/orchestrator/test_phase_error_masking.py` references `ExecutionGraph.from_plugin_instances()` or `instantiate_plugins_from_config()`. The `conftest.py` `MockCoalesceExecutor` explicitly says "Tests bypass the DAG builder" (`conftest.py:23`). CLAUDE.md's rule binds at **integration** scope; `tests/unit/engine/` is unit scope where mocks are tolerated. Flagged as a test-debt candidate qualifying that an integration-tier audit is required to determine if the production-path rule is held there — **not** a unit-test-scope defect.

## 1. engine/orchestrator/

**Path:** `src/elspeth/engine/orchestrator/`
**Responsibility:** Owns the full pipeline run lifecycle — initialisation, source loading, row dispatch into `RowProcessor`, sink writing, completion, and post-run audit export — and is the named locus of the 4-site declaration-contract dispatch shape per ADR-010 H2 (`pre_emission_check`, `post_emission_check`, `batch_flush_check`, `boundary_check`). [CITES KNOW-A25] [CITES KNOW-A26] [CITES KNOW-ADR-010i]
**Files:** 7    **LOC:** 5,173
**Type:** sub-package
**Internal coupling:**
- Sub-package `__init__.py` re-exports `Orchestrator`, `prepare_for_run`, `PipelineConfig`, `RunResult`, `RouteValidationError`, `AggregationFlushResult`, `ExecutionCounters`, `RowPlugin` from `core.py` and `types.py` (`orchestrator/__init__.py:24-32`).
- `core.py` is the entry; `types.py`, `validation.py`, `export.py`, `aggregation.py`, `outcomes.py` are focused helpers — the `__init__.py:1-22` docstring records this as a deliberate "refactored from a single 3000+ line module into focused modules while preserving the public API" decomposition.
- Drives `RowProcessor` (`engine/processor.py`), the executor suite (`engine/executors/`), and `CoalesceExecutor` (`engine/coalesce_executor.py`).

**External coupling:** Imports `contracts/` (RowResult, TokenInfo, errors), `core.config` (ElspethSettings/RuntimeRetryConfig), `core.landscape` (recorder + DataFlowRepository via TokenManager), `core.expression_parser`, `core.dependency_config`. No L0/L1 violations; layer-import conformance per L1 oracle.

**Patterns observed:** L3 deep-dive candidate (`core.py` at 3,281 LOC); internals not opened at L2 depth. The `__init__.py:14-22` module decomposition (core/types/validation/export/aggregation/outcomes) is **direct evidence of remediation effort against KNOW-A70 quality risk** — concentration here was previously a single ~3,000-line module. Public API stability is asserted ("unchanged") in the docstring.

**Concerns:** 14 defensive-pattern findings concentrated in `core.py` (R4 broad-except in `_emit_phase_error` 388, `_cleanup_plugins` 882/887/892/899/906/913, `_execute_export_phase` 1323, `Orchestrator.run` 1485/1516/1543, `Orchestrator.resume` 3156/3167/3175; R5 isinstance in `_cleanup_plugins.record_cleanup_error` 864, `_initialize_run_context` 1890, `_handle_quarantine_row` 2116; R6 silent-except in `_handle_quarantine_row` 2165). The R6 silent-except in `_handle_quarantine_row` is the highest-attention finding — quarantine-time exception swallowing has audit-trail implications. `aggregation.py:66` and `types.py:389` each have 1 R5 finding.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A26]` (Orchestrator ~3,500 LOC at orchestrator/) — verified 5,173 LOC across the package; KNOW-A26 figure is for `core.py` only. `[CITES KNOW-ADR-010i]` (4 dispatch sites). `[CITES KNOW-A70]` (large-file quality risk) — `__init__.py:14-22` is direct evidence of active remediation. `[DIVERGES FROM KNOW-A70]` insofar as KNOW-A70 cited orchestrator at "~2,070 LOC" — verified `core.py` alone is 3,281 LOC; drift has happened since ARCHITECTURE.md.
**Test surface:** `tests/unit/engine/orchestrator/test_phase_error_masking.py` (the only file in the engine unit-test tree to use `from_plugin_instances`); plus `test_aggregation.py`, `test_export.py`, `test_outcomes.py`, `test_validation.py`, `test_resume_failure.py`, `test_graceful_shutdown.py` cover the sibling helpers.
**Confidence:** **High** — sub-package shape verified via `ls`; refactoring evidence direct from `__init__.py`; deep-dive deferral respected.

---

## 2. engine/executors/

**Path:** `src/elspeth/engine/executors/`
**Responsibility:** Plugin-call wrappers that bracket each plugin invocation with audit recording, plus the post-ADR-010 declaration-contract dispatch surface — six contract adopters (one per ADR-011/012/013/014/016/017), one shared dispatcher (`declaration_dispatch.py`), one bootstrap module enforcing the closed-set adopter manifest, the structural terminal-state guard (`state_guard.py`), and the four classical plugin executors (transform, gate, sink, aggregation). [CITES KNOW-ADR-010] [CITES KNOW-ADR-010i] [CITES KNOW-ADR-009b] [CITES KNOW-A25]
**Files:** 16    **LOC:** 5,550
**Type:** sub-package
**Internal coupling:**
- Sub-package `__init__.py` re-exports `AggregationExecutor`, `GateExecutor`, `SinkExecutor`, `TransformExecutor`, `NodeStateGuard`, `GateOutcome`, `MissingEdgeError`, `AGGREGATION_CHECKPOINT_VERSION` (`executors/__init__.py:10-26`).
- `declaration_dispatch.py` is the single 4-site dispatcher (`pass_through.py:9-13`, `declared_output_fields.py:3-6`, `can_drop_rows.py:3-6`, `schema_config_mode.py:3-6` all register on `post_emission_check` + `batch_flush_check`; `declared_required_fields.py:3-5` on `pre_emission_check`; `source_guaranteed_fields.py:3-5` and `sink_required_fields.py:3-5` on `boundary_check`).
- `declaration_contract_bootstrap.py:1-11` is the closed-set authoritative import surface; `tests/unit/engine/test_declaration_contract_bootstrap_drift.py` AST-scans this directory for drift.
- `transform.py` is the principal caller of `declaration_dispatch.run_pre_emission_checks` / `run_post_emission_checks`; `pass_through.py` is also called from `RowProcessor._cross_check_flush_output` per ADR-009b.

**External coupling:** Each executor imports `contracts/` (TokenInfo, errors, RoutingMode/RowOutcome/NodeStateStatus, declaration_contracts, audit_evidence), `core.landscape` (recorder via TokenManager), `core.config` (settings types). `transform.py:15-23` imports the ADR-010 dispatch payload TypedDicts (`PostEmissionInputs`, `PreEmissionInputs`, `derive_effective_input_fields`).

**Patterns observed:** This sub-package is the highest-density `[CITES KNOW-ADR-*]` site in the engine — every contract-adopter file maps 1:1 to an accepted ADR (ADR-007/008 → `pass_through.py`; ADR-011 → `declared_output_fields.py`; ADR-012 → `can_drop_rows.py`; ADR-013 → `declared_required_fields.py`; ADR-014 → `schema_config_mode.py`; ADR-016 → `source_guaranteed_fields.py`; ADR-017 → `sink_required_fields.py`). The audit-complete-with-aggregation dispatch posture is documented inline (`declaration_dispatch.py:1-26`) and supersedes the original first-fire short-circuit.

**Concerns:** 13 defensive-pattern findings across the sub-package: `declaration_dispatch.py:137` and `:142` flag R6 silent-except for `DeclarationContractViolation` and `PluginContractViolation` — **this is in tension with the inline claim "every violation is recorded" at `declaration_dispatch.py:23-26`**; the swallowing is presumably intentional aggregation behaviour but the audit-complete claim should be verified at L3 depth. `executors/sink.py` has 5 findings (R4 broad-except in `_best_effort_cleanup` 150 and `write` 466; R5 isinstance in `write` 536/542/765/771); `executors/gate.py` has 4 (R5 268-269 in `execute_config_gate`; R1 dict.get 356/370 in `_record_routing`); `executors/transform.py` has 2 (R5 258, 440); `executors/state_guard.py` has 3 (R5 266/271, R4 275 in `_extract_audit_evidence_context`).

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-ADR-010]` (declaration-trust framework). `[CITES KNOW-ADR-010e]` (4-site nominal ABC + audit-complete dispatch — verified inline at `declaration_dispatch.py:1-15`). `[CITES KNOW-ADR-010f]` (`AggregateDeclarationContractViolation` import at `transform.py:17`). `[CITES KNOW-ADR-009b]` (`engine/executors/pass_through.py` shared by both single-token and batch-flush paths — confirmed `pass_through.py:9-13`). `[CITES KNOW-ADR-011/012/013/014/016/017]` for individual contract adopters. `[CITES KNOW-A34]` (4 plugin protocols — Transform/BatchTransform/Sink/Source executors are the audit wrappers). Test-path integrity: `[DIVERGES FROM KNOW-C44]` qualified — see Conventions test-path probe; unit-scope only.
**Test surface:** `tests/unit/engine/test_executors.py`, `test_declaration_dispatch.py`, `test_pass_through_declaration_contract.py`, `test_declared_output_fields_contract.py`, `test_declared_required_fields_contract.py`, `test_can_drop_rows_contract.py`, `test_schema_config_mode_contract.py`, `test_source_guaranteed_fields_contract.py`, `test_sink_required_fields_contract.py`, `test_boundary_dispatch_inputs.py`, `test_declaration_contract_bootstrap_drift.py`, `test_state_guard_audit_evidence_discriminator.py`, `test_sink_executor_diversion.py`, `test_failsink_validation.py` — one test file per contract module (drift-resistant pattern).
**Confidence:** **High** — every contract module mapped to an ADR via a verified docstring read; dispatcher posture verified inline; concern citations all carry file:line.

---

## 3. engine/processor.py

**Path:** `src/elspeth/engine/processor.py`
**Responsibility:** RowProcessor — the row-by-row DAG execution state machine that creates tokens, calls executors, evaluates gates, handles aggregation, and assigns terminal states; owns the second site of the pass-through cross-check (`RowProcessor._cross_check_flush_output` per ADR-009b) and the `_record_flush_violation` audit-recording path. [CITES KNOW-A25] [CITES KNOW-A26] [CITES KNOW-ADR-009b] [CITES KNOW-A44] [CITES KNOW-A45]
**Files:** 1    **LOC:** 2,700
**Type:** standalone (L3 deep-dive flag)
**Internal coupling:**
- Imports `engine.dag_navigator.DAGNavigator, WorkItem` (`processor.py:27`) — DAGNavigator is the pure topology query layer extracted from RowProcessor.
- TYPE_CHECKING-only import of `contracts.aggregation_checkpoint.AggregationCheckpointState` (`processor.py:30`).
- Called from `engine.orchestrator` (per `__init__.py:60`); transitively wires `engine.executors.*` and `engine.coalesce_executor` via the orchestrator.

**External coupling:** `contracts/` heavyweight — RouteDestination, RowOutcome, RowResult, SourceRow, TokenInfo, TransformResult, TokenRef, AuditEvidenceBase, deep_freeze, PipelineRow, BranchName/CoalesceName/NodeID/SinkName/StepResolver (`processor.py:21-26`).

**Patterns observed:** L3 deep-dive candidate; internals not opened at L2 depth. Module docstring (`processor.py:1-9`) names exactly the responsibilities cited above. `tests/unit/engine/test_processor.py:3-15` describes this as "the largest file in the engine (~2,000 lines) and the most critical path for correctness" and explicitly avoids the mock-factory anti-pattern by using `LandscapeDB.in_memory()`.

**Concerns:** 12 defensive-pattern findings — concentrated in `_handle_flush_error` (R4 broad-except 697), `_record_flush_violation` (R5 isinstance 894, R4 broad-except 927), `_route_empty_emission_results` (R4 broad-except 983), `_execute_transform_with_retry.is_retryable` (R5 1533/1535), `_record_source_boundary_failure` (R5 1604/1607, R4 broad-except 1653), `_drain_work_queue` (R5 2072), `_process_single_token` (R5 2644/2670/2674/2686). The `is_retryable` R5 isinstance check at 1533–1535 is the **retry-semantics dispatch site for cluster priority 3**.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A26]` (RowProcessor ~1,860 LOC at processor.py). `[DIVERGES FROM KNOW-A26]` — verified 2,700 LOC; the file has grown ~45% since ARCHITECTURE.md. `[CITES KNOW-ADR-009b]` (RowProcessor._cross_check_flush_output uses `engine/executors/pass_through.py`). `[CITES KNOW-A44]` (terminal-state list). `[CITES KNOW-A45]` (BUFFERED non-terminal). `[CITES KNOW-A70]` (large-file risk).
**Test surface:** `tests/unit/engine/test_processor.py`, `test_processor_pipeline_row.py`, `test_cross_check_flush_output.py`, `test_record_flush_violation_failure.py`, `test_flush_dispatcher_routing.py`, `test_row_outcome.py` (terminal-state invariant).
**Confidence:** **Medium** — responsibility cited from prior docs + docstring read only (deep-dive cap); concerns enumerated from defensive-pattern oracle; internal complexity unverified.

---

## 4. engine/coalesce_executor.py

**Path:** `src/elspeth/engine/coalesce_executor.py`
**Responsibility:** Stateful fork/join merge barrier that holds tokens correlated by `row_id` until policy conditions are met (require_all/quorum/best_effort/first), then merges row data via the configured strategy (union/nested/select); records branch-loss notifications and supports checkpoint resume. [CITES KNOW-A28] [CITES KNOW-A47] [CITES KNOW-A48] [CITES KNOW-A46]
**Files:** 1    **LOC:** 1,603
**Type:** standalone (L3 deep-dive flag)
**Internal coupling:**
- No imports from elsewhere in `engine/` at module top — self-contained relative to the engine sub-tree (`coalesce_executor.py:1-30`).
- Wired in by `orchestrator/__init__.py:24` and re-exported by `engine/__init__.py:43`.

**External coupling:** `contracts.coalesce_checkpoint`, `contracts.coalesce_enums.{CoalescePolicy, MergeStrategy}`, `contracts.coalesce_metadata`, `contracts.errors.{AuditIntegrityError, CoalesceCollisionError, CoalesceFailureReason, ContractMergeError, ExecutionError, OrchestrationInvariantError}`, `contracts.{TokenInfo}`, `structlog` for logger (`coalesce_executor.py:7-30`).

**Patterns observed:** L3 deep-dive candidate; internals not opened at L2 depth. Docstring (`coalesce_executor.py:1-5`) confirms the responsibility-claim chain to KNOW-A28/A47/A48 verbatim ("Coalesce is a stateful barrier that holds tokens until merge conditions are met. Tokens are correlated by row_id"). The size and the policy×strategy product space are **essential complexity** for the cluster-priority-1 question — the coalesce semantic surface area genuinely lives here.

**Concerns:** 4 defensive-pattern findings: `_execute_merge` R5 isinstance 1039 (`AuditIntegrityError` discriminator), R4 broad-except 1080 (cleanup); `_merge_with_original_names` R1 dict.get 1138/1141/1147 (three sites). The R1 trio is a request to make field-origin lookup contract-driven rather than dict.get-tolerant.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A28]` (CoalesceExecutor at ~1,054 LOC). `[DIVERGES FROM KNOW-A28]` — verified 1,603 LOC (~52% growth). `[CITES KNOW-A47]` (4 policies). `[CITES KNOW-A48]` (3 strategies). `[CITES KNOW-A70]` (large-file risk).
**Test surface:** `tests/unit/engine/test_coalesce_executor.py` (comprehensive — header lists all policies + strategies + timeout + flush + branch-loss + late-arrivals + audit recording per `:1-7`); `test_coalesce_contract_bug.py`, `test_coalesce_pipeline_row.py`.
**Confidence:** **Medium** — responsibility cited from docstring + prior docs; internals unverified per L3 cap.

---

## 5. engine/tokens.py

**Path:** `src/elspeth/engine/tokens.py`
**Responsibility:** TokenManager — high-level token-lifecycle interface over `core.landscape.data_flow_repository.DataFlowRepository`; supports create-from-source-row, fork-to-branches, coalesce-from-branches, and update-row-data-after-transform. [CITES KNOW-A25] [CITES KNOW-A46] [CITES KNOW-C29]
**Files:** 1    **LOC:** 399
**Type:** standalone
**Internal coupling:** None visible at module top; called from `RowProcessor` (`processor.py` references `TokenInfo` from `contracts/`) and re-exported by `engine/__init__.py:63`.
**External coupling:** `contracts.{SourceRow, TokenInfo}`, `contracts.audit.TokenRef`, `contracts.errors.OrchestrationInvariantError`, `contracts.schema_contract.{PipelineRow, SchemaContract}`, `contracts.types.{NodeID, StepResolver}`, **`core.landscape.data_flow_repository.DataFlowRepository`** (`tokens.py:14-19`). The cross-layer dependency to `core.landscape` is the persistence side of token identity.

**Patterns observed:** Per the docstring (`tokens.py:1-5`), this is "a simplified interface over DataFlowRepository for managing tokens". This is the **engine-side orchestration locus** for token identity (cluster priority 2), but **token identity is not solely owned here**: (a) `tokens.py` orchestrates create/fork/coalesce/update; (b) `core.landscape.data_flow_repository` persists; (c) fork/join call sites live in `processor.py` and `orchestrator/core.py` (per the deep-dive flag's docstring on row processing). Three loci, not one.

**Concerns:** 1 defensive-pattern finding: `TokenManager.create_quarantine_token` R5 isinstance 156 (`isinstance(source_row.row, dict)`). This is a Tier-3 boundary check (source row data of unknown shape during quarantine) — likely legitimate but should carry an allowlist entry.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A25]` (TokenManager listed in engine components). `[CITES KNOW-A46]` (row_id stable / token_id unique / parent_token_id lineage). `[CITES KNOW-C29]` (Token identity tracks instances through forks/joins).
**Test surface:** `tests/unit/engine/test_tokens.py` (`tokens.py:1` test header confirms scope); `test_token_manager_pipeline_row.py`; `test_batch_token_identity.py`.
**Confidence:** **High** — file size manageable; docstring + import section read; cross-layer coupling to DataFlowRepository explicit at line 19.

---

## 6. engine/triggers.py

**Path:** `src/elspeth/engine/triggers.py`
**Responsibility:** TriggerEvaluator for aggregation batches — combines configured `count`, `timeout`, `condition` triggers with OR semantics (first-to-fire wins) and records which trigger fired for audit; `end_of_source` is implicit and engine-handled. [CITES KNOW-A25] [CITES KNOW-P14]
**Files:** 1    **LOC:** 324
**Type:** standalone
**Internal coupling:** Imports `engine.clock.DEFAULT_CLOCK` (`triggers.py:25`) and `engine.clock.Clock` under TYPE_CHECKING (`triggers.py:27-28`).
**External coupling:** `contracts.enums.TriggerType`, `core.config.TriggerConfig`, `core.expression_parser.ExpressionParser` (`triggers.py:22-24`).

**Patterns observed:** Trigger semantics documented inline (`triggers.py:1-15`) as "Multiple triggers can be combined (first one to fire wins)". Time abstraction via `Clock` (DI pattern) supports testable timeout-dependent code paths.

**Concerns:** 2 defensive-pattern findings: `TriggerEvaluator.record_accept` R5 isinstance 129 and `TriggerEvaluator.should_trigger` R5 isinstance 185 — both are `isinstance(result, bool)` against expression-parser return values, i.e., guarding against a non-bool from a Tier-3 operator-authored expression (`condition` trigger). These are legitimate Tier-3 boundary checks; should carry allowlist entries.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A25]` (Triggers listed in engine components). `[CITES KNOW-P14]` (trigger.count + output_mode in aggregation YAML).
**Test surface:** `tests/unit/engine/test_triggers.py`.
**Confidence:** **High** — small file, fully read at module head; trigger types match institutional docs.

---

## 7. engine/dag_navigator.py

**Path:** `src/elspeth/engine/dag_navigator.py`
**Responsibility:** Pure topology queries for DAG traversal — extracted from RowProcessor to create a clean service boundary; resolves jump targets, creates continuation work items, classifies fork origins. [CITES KNOW-A25] [CITES KNOW-A67] [CITES KNOW-C28]
**Files:** 1    **LOC:** 316
**Type:** standalone
**Internal coupling:** Forward-references `engine.orchestrator.types.RowPlugin` and `engine.processor.DAGTraversalContext` under TYPE_CHECKING (`dag_navigator.py:25-27`); imported by `processor.py:27`.
**External coupling:** `contracts.TransformProtocol`, `contracts.errors.OrchestrationInvariantError`, `contracts.types.{CoalesceName, NodeID}`, `core.config.GateSettings` (`dag_navigator.py:19-22`).

**Patterns observed:** Module docstring (`dag_navigator.py:1-10`) names the extraction explicitly: "Pure topology queries for DAG traversal. Extracted from RowProcessor to create a clean service boundary." Methods are described as "pure queries on immutable topology data — no mutable state dependencies." This is **direct evidence of remediation** in the same vein as the orchestrator decomposition.

**Concerns:** 3 defensive-pattern findings: `resolve_jump_target_sink` R5 isinstance 219 (`GateSettings`), 221 (`TransformProtocol` + `on_success is not None`); `create_continuation_work_item` R5 isinstance 277 (`GateSettings`). These are protocol-discrimination at a config-vs-plugin boundary; the trust-tier rule normally says protocol checks are bug-masking, but at config dispatch this is type-narrowing of a heterogeneous registry. Allowlist-with-justification candidate.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A25]` (DAGNavigator listed). `[CITES KNOW-A67]` (ADR-005 declarative DAG wiring — every edge explicitly declared). `[CITES KNOW-C28]` (linear pipelines as degenerate DAGs).
**Test surface:** `tests/unit/engine/test_dag_navigator.py`.
**Confidence:** **High** — file size manageable, extraction rationale verified inline.

---

## 8. engine/batch_adapter.py

**Path:** `src/elspeth/engine/batch_adapter.py`
**Responsibility:** SharedBatchAdapter — bridges `TransformExecutor` to batch-aware transform worker pools, registering one `RowWaiter` per (token_id, state_id) so multiple in-flight rows can share a single batch transform output port without leaking results to retry attempts. [CITES KNOW-P13] [CITES KNOW-A34]
**Files:** 1    **LOC:** 284
**Type:** standalone
**Internal coupling:** Used by `engine.executors.transform.TransformExecutor` (per `batch_adapter.py:7-9` architecture comment).
**External coupling:** Not visible in the first 30 lines (docstring-heavy header); the architecture comment names the call chain `Orchestrator → TransformExecutor → [BatchTransformMixin] → Worker Pool → SharedBatchAdapter`.

**Patterns observed:** Inline architecture diagram (`batch_adapter.py:6-19`) documents the (token_id, state_id) keying as a **retry-safety invariant** — stale results from timed-out workers go to garbage-collected waiters, not to the retry's new waiter (`batch_adapter.py:21-25`). Plugin exception propagation goes through `ExceptionResult` to satisfy the CLAUDE.md crash-on-plugin-bug rule (`batch_adapter.py:27-30`).

**Concerns:** 2 defensive-pattern findings: `RowWaiter.wait` R9 dict.pop-default 124 (`self._entries.pop(self._key, None)`) and R5 isinstance 136 (`isinstance(entry.result, ExceptionResult)`). Both are at the result-routing boundary; the dict.pop-default is suspect (silent missing-key tolerance) but may be the documented retry-stale-waiter cleanup pattern.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-P13]` (batch-aware transforms with `is_batch_aware=True`). `[CITES KNOW-A34]` (4 plugin protocols including BatchTransform).
**Test surface:** `tests/unit/engine/test_batch_adapter.py`.
**Confidence:** **High** — architecture documented inline; concerns sourced from oracle.

---

## 9. engine/spans.py

**Path:** `src/elspeth/engine/spans.py`
**Responsibility:** SpanFactory — OpenTelemetry span creation for pipeline execution; provides a static-name span hierarchy (`run` → `source`/`row` → `transform`/`sink`/`aggregation`) with plugin identity carried in attributes; falls back to `NoOpSpan` when no tracer is configured. [CITES KNOW-A25] [CITES KNOW-A69]
**Files:** 1    **LOC:** 279
**Type:** standalone
**Internal coupling:** None at module top; consumed by `processor.py` and `orchestrator/core.py` (per `engine/__init__.py:62` re-export).
**External coupling:** TYPE_CHECKING-only `opentelemetry.trace.{Span, Tracer}` (`spans.py:19-20`); the runtime fallback is a pure-Python `NoOpSpan`.

**Patterns observed:** Module docstring (`spans.py:5-12`) records the static-name decision: "span names are static; plugin identity is in attributes" — explicit anti-pattern guard against high-cardinality span names. `NoOpSpan` is the no-tracer-configured codepath.

**Concerns:** None observed in the defensive-pattern oracle for this file.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A25]` (SpanFactory listed in engine components). `[CITES KNOW-A69]` (OpenTelemetry as implicit tech choice).
**Test surface:** `tests/unit/engine/test_spans.py`.
**Confidence:** **High** — clean, well-documented small file.

---

## 10. engine/dependency_resolver.py

**Path:** `src/elspeth/engine/dependency_resolver.py`
**Responsibility:** Pipeline-dependency resolution — cycle detection, depth limiting, and execution of `depends_on` sub-pipelines for the bootstrap/preflight phase. [CITES KNOW-C28] [CITES KNOW-A25]
**Files:** 1    **LOC:** 173
**Type:** standalone
**Internal coupling:** None at module top; `bootstrap.resolve_preflight` is the orchestrator-side caller (per `bootstrap.py:24`).
**External coupling:** `contracts.errors.{DependencyFailedError, GracefulShutdownError}`, `contracts.pipeline_runner.PipelineRunner`, `contracts.enums.RunStatus`, `core.canonical.canonical_json`, `core.dependency_config.{DependencyConfig, DependencyRunResult}`, `pydantic.ValidationError`, `yaml` (`dependency_resolver.py:10-18`).

**Patterns observed:** `_load_depends_on` is explicitly framed as a **Tier-3 boundary read** (`dependency_resolver.py:28` "validates structure of operator-authored YAML") — i.e., this is one of the few engine-tree files that legitimately operates at zero-trust input boundary.

**Concerns:** 4 defensive-pattern findings, ALL in `_load_depends_on`: R5 isinstance 39, R1 dict.get 45, R5 isinstance 46, R5 isinstance 51. **These findings are arguably correct-as-written** — the function's docstring at line 28 names this as a Tier-3 boundary, where coercion-aware validation is permitted per CLAUDE.md trust-tier rules. Strong allowlist candidate with `reason="Tier-3 boundary: operator YAML"`.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-C28]` (pipelines compile to DAGs). `[CITES KNOW-A25]` (engine components include dependency resolution semantics).
**Test surface:** `tests/unit/engine/test_dependency_resolver.py`.
**Confidence:** **High** — small file, fully read at module head; concerns oracle-derived.

---

## 11. engine/retry.py

**Path:** `src/elspeth/engine/retry.py`
**Responsibility:** RetryManager — tenacity-based retry wrapper for transform execution; exponential backoff with jitter, max-attempts, retryable-error filtering, and per-attempt audit hooks via `on_retry` callback. [CITES KNOW-A25] [CITES KNOW-A69]
**Files:** 1    **LOC:** 137
**Type:** standalone
**Internal coupling:** None at module top; called from `processor.py._execute_transform_with_retry` (the R5 isinstance findings at `processor.py:1533-1535` enumerate the retryable-error set).
**External coupling:** `tenacity.{RetryCallState, RetryError, …}`; `contracts.config.RuntimeRetryConfig` and `contracts.errors.MaxRetriesExceeded` per `engine/__init__.py:36-37` re-exports.

**Patterns observed:** Module docstring (`retry.py:9-15`) names the audit-integration contract: "Each retry attempt must be auditable with the key (run_id, row_id, transform_seq, attempt). The on_retry callback should call recorder.record_retry_attempt() to audit each attempt, ensuring complete traceability of transient failures and recovery." This is one half of the cluster-priority-3 question — RetryManager owns the **execution loop**; `processor.py` owns the **terminal-state assignment** (FAILED if retries exhausted) and the **retryable-error classification** (`processor.py:1533-1535`). Two-locus split.

**Concerns:** None observed in the defensive-pattern oracle for `retry.py`. (The R5 isinstance findings flagged at `processor.py:1533/1535` belong to the processor entry, but they are the dispatch site that decides what `retry.py` re-runs.)

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A25]` (RetryManager listed). `[CITES KNOW-A69]` (tenacity as implicit tech choice).
**Test surface:** `tests/unit/engine/test_retry.py` (`retry.py:1-2` confirms scope), `test_retry_policy.py`, `test_plugin_retryable_error.py`.
**Confidence:** **High** — small file, audit-integration contract documented inline.

---

## 12. engine/commencement.py

**Path:** `src/elspeth/engine/commencement.py`
**Responsibility:** Pre-flight commencement-gate evaluation — assembles a sandboxed expression-evaluation context (`dependency_runs`, `collections`, `env`) and runs operator-authored go/no-go expressions before the run starts; raises `CommencementGateFailedError` on no-go. [CITES KNOW-A25] [CITES KNOW-C7]
**Files:** 1    **LOC:** 137
**Type:** standalone
**Internal coupling:** None at module top; called from `bootstrap.resolve_preflight` (per `bootstrap.py:14-17`).
**External coupling:** `contracts.errors.CommencementGateFailedError`, `contracts.freeze.deep_freeze`, `core.dependency_config.{CommencementGateConfig, CommencementGateResult}`, `core.expression_parser.ExpressionParser` (`commencement.py:9-12`).

**Patterns observed:** `_GATE_ALLOWED_NAMES = ["collections", "dependency_runs", "env"]` (`commencement.py:14`) is a closed-list namespace whitelist for the expression sandbox — a deliberate constraint on operator-authored expressions. Docstring describes this as "pre-flight go/no-go checks."

**Concerns:** 1 defensive-pattern finding: `evaluate_commencement_gates` R5 isinstance 94 (`isinstance(result, bool)`). Same rationale as `triggers.py` — guards against a non-bool from operator-authored expression at a Tier-3 boundary; allowlist candidate.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A25]` (commencement gates as engine surface). `[CITES KNOW-C7]` (auditability standard — gate decisions are recorded as preconditions).
**Test surface:** `tests/unit/engine/test_commencement.py`.
**Confidence:** **High** — small file, complete header read.

---

## 13. engine/bootstrap.py

**Path:** `src/elspeth/engine/bootstrap.py`
**Responsibility:** Programmatic pipeline bootstrap — `resolve_preflight()` is the shared codepath for CLI and programmatic callers, running dependency-resolution and commencement-gate evaluation if configured. [CITES KNOW-A25] [CITES KNOW-C45]
**Files:** 1    **LOC:** 138
**Type:** standalone
**Internal coupling:** Imports `core.dependency_config.{CommencementGateResult, DependencyRunResult, PreflightResult}` (`bootstrap.py:11-15`); orchestrates `dependency_resolver` and `commencement` (per `bootstrap.py:24-30` docstring).
**External coupling:** `contracts.errors.FrameworkBugError`, `contracts.pipeline_runner.PipelineRunner`, `contracts.probes.CollectionProbe`, `core.config.ElspethSettings`.

**Patterns observed:** The docstring (`bootstrap.py:24-30`) names the design rationale: "Extracted so both the CLI and programmatic callers share the same codepath" — a single-source-of-truth pattern preventing CLI/programmatic drift.

**Concerns:** None observed in the defensive-pattern oracle for this file.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A25]` (engine components include preflight). `[CITES KNOW-C45]` (config precedence — preflight reads ElspethSettings post-merge).
**Test surface:** `tests/unit/engine/test_bootstrap_preflight.py`.
**Confidence:** **High** — small file, design rationale explicit in docstring.

---

## 14. engine/clock.py

**Path:** `src/elspeth/engine/clock.py`
**Responsibility:** `Clock` protocol abstraction over `time.monotonic()` enabling deterministic testing of timeout-dependent code (aggregation triggers, coalesce timeouts); provides `SystemClock` (production) and `MockClock` (test) implementations with `DEFAULT_CLOCK` constant. [CITES KNOW-A28]
**Files:** 1    **LOC:** 121
**Type:** standalone
**Internal coupling:** Consumed by `triggers.py` (`triggers.py:25,27-28`) and (per docstring) `coalesce_executor.py` for timeout logic.
**External coupling:** `typing.Protocol` only (`clock.py:15`); minimal surface area.

**Patterns observed:** A textbook DI/Protocol abstraction with no `Concerns` and a clear test/production split documented inline (`clock.py:21-26`).

**Concerns:** None observed in the defensive-pattern oracle for this file.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A28]` (CoalesceExecutor uses Clock for timeout — verified by docstring at `clock.py:5`).
**Test surface:** `tests/unit/engine/test_clock.py`.
**Confidence:** **High** — small leaf module, no concerns.

---

## 15. engine/__init__.py

**Path:** `src/elspeth/engine/__init__.py`
**Responsibility:** Public-API surface of the engine subsystem — re-exports the orchestrator suite (Orchestrator, PipelineConfig, RunResult, RouteValidationError, AggregationFlushResult, ExecutionCounters, RowPlugin), the executor suite (Aggregation/Gate/Sink/Transform), CoalesceExecutor + CoalesceOutcome, RowProcessor, RetryManager, SpanFactory, TokenManager, and the ExpressionParser surface; stable contract for L3 callers. [CITES KNOW-A25] [CITES KNOW-A26]
**Files:** 1    **LOC:** 91
**Type:** sub-subsystem (public-API surface)
**Internal coupling:** Imports from every sub-package and standalone module of `engine/` except `bootstrap.py`, `clock.py`, `commencement.py`, `dag_navigator.py`, `dependency_resolver.py`, `triggers.py`, and `batch_adapter.py` — i.e., these seven are **not part of the public API** and are accessed only through internal call paths or via dotted imports by callers who know where to find them. (`engine/__init__.py:35-91`)
**External coupling:** Re-exports `contracts.{RowResult, TokenInfo}`, `contracts.config.RuntimeRetryConfig`, `contracts.errors.MaxRetriesExceeded`, `core.expression_parser.{ExpressionParser, ExpressionSecurityError, ExpressionSyntaxError}` (`engine/__init__.py:35-42`).

**Patterns observed:** The closed `__all__` list (`engine/__init__.py:65-91`) is alphabetised and explicitly enumerates 25 names. Notable **non-exports**: `DAGNavigator`, `Clock`, `TriggerEvaluator`, `SharedBatchAdapter`, `resolve_preflight`, `evaluate_commencement_gates` — these are engine-internal contracts, not L3 surface. The example block (`__init__.py:11-32`) demonstrates the canonical entry shape using `ExecutionGraph.from_plugin_instances(...)` per CLAUDE.md's test-path-integrity rule.

**Concerns:** None observed in the defensive-pattern oracle for this file.

**L1 cross-reference:** Supplements 02-l1-subsystem-map.md §3 (engine/).
**[CITES] / [DIVERGES FROM]:** `[CITES KNOW-A25]` (engine components list — verified by `__all__`). `[CITES KNOW-A26]` (Orchestrator + RowProcessor are the headline exports). `[CITES KNOW-C44]` (the docstring example at lines 11-32 demonstrates `ExecutionGraph.from_plugin_instances()` per the test-path integrity rule, anchoring the production-path contract at the engine boundary).
**Test surface:** Public-API stability is asserted by the existence of `engine/__init__.py:65-91` `__all__` list; no dedicated test file under `tests/unit/engine/` (any breakage would surface via the per-file unit tests). Invariant asserted in code only; no test coverage observed at L2 depth — L2 debt candidate (no public-API surface test).
**Confidence:** **High** — file fully read.

---

## Closing

### L3-deep-dive candidates flagged at L2 (3 files)

| File | LOC | Cluster priority addressed |
|------|-----|----------------------------|
| `engine/orchestrator/core.py` | 3,281 | Priority 1 (concentration); Priority 2 (token-identity orchestration is split, fork/join call sites here); Priority 3 (terminal-state cleanup paths in `_cleanup_plugins` + `Orchestrator.run`) |
| `engine/processor.py` | 2,700 | Priority 1 (concentration); Priority 2 (fork/join + token-identity dispatch); Priority 3 (retry classification at `_execute_transform_with_retry.is_retryable`); Priority 4 (terminal-state assignment) |
| `engine/coalesce_executor.py` | 1,603 | Priority 1 (concentration — but essential per policy×strategy product); Priority 2 (coalesce branch of token identity) |

### Test-debt candidates surfaced (per Δ L2-5)

1. **Test-path integrity probe** (entry 2 cross-ref + Conventions): only `tests/unit/engine/orchestrator/test_phase_error_masking.py` references `from_plugin_instances` / `instantiate_plugins_from_config` in the engine unit-test tree; integration-tier audit required to verify the CLAUDE.md rule binds. This is **not** a unit-test defect — flagging the absence of an integration check from this catalog's vantage.
2. **Public-API surface test for `engine/__init__.py:__all__`** (entry 15): the alphabetised closed list of 25 names has no dedicated stability test; per-file unit tests would only surface breakage indirectly.
3. **`declaration_dispatch.py:137,142` R6 silent-except vs audit-complete claim** (entry 2): the inline claim "every violation is recorded" should be locked by a test asserting both `DeclarationContractViolation` and `PluginContractViolation` raised from registered contracts arrive in the dispatcher's aggregation list (vs being swallowed) — `tests/unit/engine/test_declaration_dispatch.py` is the candidate but content not verified at L2 depth.

### Cross-cluster observations bookmarked for synthesis

(One-line each; report wave to integrate. Not in entries per Δ L2-3.)

- **engine ↔ core.landscape coupling at `tokens.py:19`:** TokenManager is a thin façade over `DataFlowRepository` — engine's audit-side persistence is fully delegated to core. (Cross-cluster: engine + core.)
- **engine ↔ contracts ADR-010 dispatch payloads:** `executors/transform.py:16-23` imports the entire ADR-010 declaration-contracts payload typedict surface from `contracts.declaration_contracts`. (Cross-cluster: engine + contracts.)
- **engine ↔ core.expression_parser at three sites:** `triggers.py:24`, `commencement.py:12`, and re-exported through `engine/__init__.py:38-42`; expression evaluation is engine-consumed but core-owned. (Cross-cluster: engine + core.)
- **engine ↔ contracts.pipeline_runner protocol:** `bootstrap.py:8` and `dependency_resolver.py:14` both consume `PipelineRunner`; the runner is contracts-defined but engine-implemented at orchestration scope. (Cross-cluster: engine + contracts.)

### The 4 cluster priorities — answers

1. **KNOW-A70 essential vs accidental complexity at orchestrator/processor/coalesce.** **Mixed.** The `orchestrator/__init__.py:1-22` docstring is direct evidence that the orchestrator's previous 3,000-line single-module form has been **actively decomposed into 6 focused modules** (`core.py`, `types.py`, `validation.py`, `export.py`, `aggregation.py`, `outcomes.py`) — that is remediated accidental complexity. `coalesce_executor.py` is **essential** complexity: the policy×strategy product space (4 policies × 3 strategies + branch-loss + late-arrivals + checkpoint resume) genuinely lives there per `coalesce_executor.py:1-5` and KNOW-A47/A48. `processor.py` (2,700 LOC standalone) shows **no L2-visible decomposition**; whether its size is essential or accidental cannot be answered without an L3 deep-dive.
2. **Token identity locus.** **Three-locus split.** (a) `engine/tokens.py` orchestrates create/fork/coalesce/update via DataFlowRepository (`tokens.py:1-5,19`); (b) `core.landscape.data_flow_repository` (out of cluster) owns persistence; (c) fork/join call sites live in `engine/processor.py` and `engine/orchestrator/core.py` — both deep-dive flagged. The 399-LOC tokens.py file is the **engine-side façade**, not the sole owner.
3. **Retry semantics + terminal-state guarantee.** **Two-locus split.** `engine/retry.py` owns the tenacity retry loop and audit hook contract (`retry.py:9-15`); `engine/processor.py` owns retryable-error classification (`processor.py:1533-1535` is the dispatch site) and terminal-state assignment. The terminal-state invariant ("every row reaches exactly one terminal state") is structurally enforced by `engine/executors/state_guard.py:1-10` (`NodeStateGuard` — context-manager-as-invariant pattern); `tests/unit/engine/test_row_outcome.py` locks `RowOutcome` enum semantics on `RowResult`; `tests/unit/engine/test_state_guard_audit_evidence_discriminator.py` locks the guard's audit-evidence behaviour. **The structural-guarantee story is genuinely good; deep-dive into `processor.py` would confirm.**
4. **Test-path integrity rule (CLAUDE.md Δ L2-5).** Only one engine unit test file uses `from_plugin_instances`; engine unit tests broadly mock heavily (per `conftest.py:23`'s explicit "Tests bypass the DAG builder" comment on `MockCoalesceExecutor`). The CLAUDE.md rule binds at **integration** scope, not unit; `tests/unit/engine/` is unit-scope where mocks are tolerated. Flagged as test-debt-conditional pending an integration-tier audit (no integration test directory examined here per cluster scope).

---

**Confidence distribution:** High = 12, Medium = 3 (entries 3, 4, 1 per deep-dive constraint or LOC-drift uncertainty), Low = 0.
