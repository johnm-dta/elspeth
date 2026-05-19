# I-1 — Raw Agent Reports

Verbatim outputs from the 5 specialist agents that reviewed I-1 on 2026-05-06.
Preserved in full as the source of truth backing the synthesis in `i-1-findings.md`.

Agent IDs (not durable across sessions):
- test-suite-reviewer: `ab1bb33693b6bc428`
- quality-assurance-analyst: `a63ac58b53b88af95`
- python-code-reviewer: `a8dc22967c0a52188`
- pr-test-analyzer: `ac7c46075186b181b`
- coverage-gap-analyst: `a3e87fec7a60221da`

(Per-agent reports follow; see synthesis for cross-cutting patterns.)

---

## 1. ordis-quality-engineering:test-suite-reviewer

**Summary**

11 findings across 4 files, with a dominant pattern of skeletal construction and copy-paste fixture proliferation. `test_fixes.py` is an unrefactored regression dumping ground containing tautologies, skips production code paths, and tests language-level properties rather than audit behaviour. `test_error_persistence.py` contains two manual raw SQL token inserts that bypass the recorder API. The 9 separate `*_serialization_roundtrip.py` files are structurally identical and duplicate ~100 lines of helper code each. Overall verdict: **Mixed**.

### Findings

| File:Line | Category | Severity | Rationale | Recommendation |
|---|---|---|---|---|
| `test_fixes.py:25-56` | Tautology + production-path bypass | Major | `test_full_plugin_discovery_flow` counts plugins then does `source.node_id = "node-123"; assert source.node_id == "node-123"`. No pipeline run, no Landscape write. | Delete or replace with a test that runs a pipeline and queries the audit trail. |
| `test_fixes.py:133-147` | Tautology | Major | `test_edge_info_immutability` constructs frozen dataclass and asserts that `frozen=True` raises `AttributeError`. Tests Python's dataclass machinery. | Delete. The freeze-guard CI script covers this class of correctness at build time. |
| `test_fixes.py:149-173` | Tautology + production-path bypass | Major | `test_routing_mode_is_enum_throughout_dag` constructs `ExecutionGraph()` directly with manual `add_node`/`add_edge` then asserts `isinstance(edge.mode, RoutingMode)`. CLAUDE.md: invalid for an integration test. | Delete. |
| `test_fixes.py:175-211` | Tautology | Major | `test_plugin_node_id_on_all_plugin_types` sets `plugin.node_id = X` on three plugins and asserts `plugin.node_id == X`. | Delete. |
| `test_fixes.py:58-77` | Production-path bypass | Major | `test_dag_uses_typed_edges` manually constructs `ExecutionGraph()` via `add_node`/`add_edge` instead of `from_plugin_instances()`. | Rewrite via `tests/fixtures/pipeline`. |
| `test_fixes.py:105-131` and `213-241` | "Integration" test that is unit | Minor | `test_plugin_context_recorder_can_record` and `test_landscape_recorder_run_lifecycle` only call `begin_run` / `complete_run` with no rows, no tokens, no pipeline execution. | Merge into `test_recorder_runs.py`. |
| `test_error_persistence.py:161-175` | Production-path bypass | Critical | Raw `conn.execute(tokens_table.insert().values(token_id="token-123", ...))` with comment "to match test expectations." Bypasses `factory.data_flow.create_token()`. | Replace with `factory.data_flow.create_token(row_id=row.row_id)`. |
| `test_error_persistence.py:246-260` | Production-path bypass | Critical | Same pattern, `token_id="token-456"`. | Same fix. |
| `test_fixes.py` (whole file) | Regression dumping ground | Major | File named "fixes," header cites "Tasks 1-7," body mixes DAG unit tests with lifecycle plumbing checks and tautologies. | Redistribute survivors; delete the file. |
| `test_recorder_calls.py:1`, `test_recorder_contracts.py:1`, `test_recorder_nodes.py:1`, `test_recorder_row_data.py:1`, `test_recorder_runs.py:1` | Stale path comment | Minor | Five files open with `# tests/core/landscape/...` (pre-migration path). | Update to `# tests/integration/audit/...`. |
| `test_contract_audit.py:621-635` | Defensive read pattern | Minor | `if restored1: ...` and `if restored2: ...` guard. Tier-1 data: if `get_run_contract` returns `None` after we just stored a contract, that is corruption and must crash. | Remove the `if` guards. |

### Cross-File Patterns

- **9-file copy-paste cluster.** Each defines identical `_setup_landscape`, `_record_failure`, `_contract`, `_row`, `_plugin` helpers. Belongs in one parametrized class in `tests/integration/audit/recorder/test_declaration_contract_roundtrip.py`. Schema change to `_record_failure` requires 9 identical edits.
- **Categorisation mismatch in `test_recorder_*.py`.** Stale path headers confirm migration from `tests/core/landscape/`. They are repository tests — real DB writes and reads — correctly here for their stated scope (factory API), but reviewers should not confuse them with audit-guarantee integration tests that run the full engine.
- **Good pattern worth calling out.** `test_audit_field_separation.py` is the model: uses `ExecutionGraph.from_plugin_instances()`, mocks only the external Azure boundary, asserts both sink output and `success_reason_json` column.
- **`test_fixes.py` task tagging in comments.** Lines cite "Task 1", "Task 4", "Task 5". Sprint-internal references that should have been removed on merge.

### Top 5 Deletion Candidates

1. `test_fixes.py:133-147` — tests Python's `frozen=True`.
2. `test_fixes.py:175-211` — three-way assignment tautology.
3. `test_fixes.py:25-56` — plugin count + assignment tautology, no Landscape write.
4. `test_fixes.py:149-173` — asserts type of values the test inserted.
5. `test_fixes.py` (entire file).

### Production-Code-Path Bypass Findings

| File:Line | Bypass |
|---|---|
| `test_fixes.py:58-77` | `ExecutionGraph()` constructed with `add_node`/`add_edge` directly. |
| `test_error_persistence.py:161-175` | Raw `tokens_table.insert()` bypasses `factory.data_flow.create_token()`. |
| `test_error_persistence.py:246-260` | Same raw insert pattern. |

The `_plugin = type("X", (), {})()` bare-object helpers in roundtrip files are NOT a bypass — those tests exercise the declaration dispatcher and repository serialisation, not the full SDA pipeline.

### Out-of-Scope Observations

- `test_contract_audit.py` `MockContext` should be promoted to `tests/fixtures/`.
- `test_recorder_explain.py:27` imports from another test file's internal helpers — should live in `tests/fixtures/`.

**Confidence:** High. **Risk:** Two Critical findings carry low production risk (real DB paths) but high maintenance risk. **Information Gaps:** `tests/integration/audit/conftest.py` not exhaustively read.

---

## 2. axiom-sdlc-engineering:quality-assurance-analyst

### Verdict

**Theatre score: Pervasive.** The dominant pattern is **"integration tests that bypass the integration boundary"**: 26 of 32 files in this chunk never instantiate `Orchestrator`, `ExecutionGraph.from_plugin_instances()`, or `instantiate_plugins_from_config()`, despite being placed under `tests/integration/audit/`. They call `RecorderFactory` repos directly, manually fabricate the violation production would compute, and assert it round-trips through SQLAlchemy. The 9-file `*_serialization_roundtrip.py` cluster is the most acute manifestation: 7 of them are ~85% identical scaffolding wrapped around six different contract names.

### Theatre findings

| File:line | Category | Severity | Rationale |
|---|---|---|---|
| `test_declared_output_fields_serialization_roundtrip.py:189-232` | Test re-implements SUT | High | Test catches the violation in a `try/except`, manually constructs an `ExecutionError` from `violation.to_audit_dict()`, writes it to `error_json`, reads it back, and asserts the value equals what was just put in. |
| `test_declared_required_fields_serialization_roundtrip.py:` whole-file | Copy-paste duplicate | High | 90% identical to `test_declared_output_fields_*` per `diff` (40 changed lines out of ~400). |
| `test_source_guaranteed_fields_serialization_roundtrip.py:` whole-file | Copy-paste duplicate | High | Same scaffolding, BoundaryInputs flavour. |
| `test_sink_required_fields_serialization_roundtrip.py:` whole-file | Copy-paste duplicate | High | Same scaffolding, sink boundary flavour. |
| `test_schema_config_mode_serialization_roundtrip.py:` whole-file | Copy-paste duplicate | High | Same scaffolding, SchemaConfigMode flavour. |
| `test_declaration_contract_landscape_serialization_roundtrip.py:189+` | Round-trip-only theatre + mock-tautology | High | Tests assert values placed into `error.context` come back from `error_json`. No assertion that production's dispatch site is invoked from the engine. |
| `test_recorder_artifacts.py:17-58` | Round-trip-only theatre | Medium | `register_artifact(content_hash="abc123")` then `assert artifact.path_or_uri == "/output/result.csv"`. `content_hash` is never asserted, never compared to `stable_hash(payload)`. |
| `test_recorder_artifacts.py:113-178` | Frozen-snapshot | Medium | Asserts idempotency by re-registering with same key — but `content_hash="abc123"` is a literal string, not derived from the artifact's content. |
| `test_recorder_runs.py:20-77` | Mock-tautology | High | `test_begin_run`, `test_complete_run_*`, `test_get_run` all do `factory.x → assert x.field == value-just-passed`. |
| `test_recorder_runs.py:171-280` | Genuine | — | Field-resolution tier-1 crash tests are real audit-integrity assertions. **Keep.** |
| `test_fixes.py` whole file | Regression-dump misnamed | High | None of the 8 tests are *audit* tests. |
| `test_recorder_calls.py:60-98, 351-362` | Hash-without-binding | Medium | `assert call.request_hash is not None` / `is None`. Not bound to `stable_hash(input)`. Already partly fixed at `:127-165`. |
| `test_recorder_calls.py:319-349` | Mock-tautology | Medium | `latency_ms=None` → assert `call.latency_ms is None`. Pure ORM round-trip. |
| `test_not_null_constraints.py:31-200` | Genuine | — | Tier-1 NOT NULL enforcement at DB schema level. **Keep.** |
| `test_tier1_integrity.py:75-280` | Genuine | — | Real Tier-1 crash-on-anomaly assertions. **Keep, use as template.** |
| `test_source_boundary_orchestrator.py:18-52` | Genuine | — | Only orchestrator-driven test in the chunk that uses `Orchestrator(db).run(...)` against a production-shape pipeline AND queries the recorder for the failure shape. **Use as template.** |
| `test_can_drop_rows_roundtrip.py:` | Mixed | Low | DOES build an `ExecutionGraph` and run via `Orchestrator`. Closer to a real integration. |
| `test_recorder_routing_events.py` | Mixed | — | Uses `Orchestrator`. Keep as a model. |
| `test_export.py:90-208` | Genuine | — | Loads YAML settings, runs via CLI/Orchestrator, asserts export artifact exists and is signable. |
| `test_export.py:230-460` | Suspicious | Medium | Signed-export hash determinism may be testing rfc8785 rather than the export pipeline. |
| `test_pass_through_violation_persists.py:1-50` | Honest | — | Docstring explicitly justifies SQL-level `json_extract`. Keep. |
| `test_recorder_explain.py:36-410` | Mostly genuine | — | Multiple `explain()` lineage tests with payload_store binding. **Keep.** |
| `test_audit_field_separation.py` | Likely genuine | — | Uses production path. |
| `test_exporter_batch_queries.py` | Likely genuine | — | Uses production path. |
| `test_sqlcipher_pipeline.py` | Smoke test | Low | Small. Encrypted backend smoke check. |

### 9-file roundtrip cluster analysis

**Cluster:** `test_declaration_contract_landscape_serialization_roundtrip.py` (557), `test_declared_output_fields_*` (401), `test_declared_required_fields_*` (362), `test_source_guaranteed_fields_*` (365), `test_sink_required_fields_*` (386), `test_schema_config_mode_*` (345), `test_can_drop_rows_roundtrip.py` (434, outlier — uses Orchestrator).

What they share verbatim: `_setup_landscape()`, `_record_failure()` (~30 lines), `setup_method`/`teardown_method` snapshot/restore, the try/except violation→ExecutionError pattern, the `assert "sk-abcdef" not in json.dumps(context)` secret-redaction tail (5 verbatim copies), TypedDict `_RoundTripPayload`.

What differs: contract class, dispatch site, field literal sets, violation class name.

**Consolidation:** ~2,400 lines → ~400 lines. Use `test_can_drop_rows_roundtrip.py` and `test_source_boundary_orchestrator.py` as the template. Each contract should have ONE orchestrator-driven case + parametrized table for audit-payload-shape assertions. Secret-redaction tail moves to single `test_secret_redaction.py`.

### `make_recorder_with_run` boilerplate (~15 lines) repeated in nearly every recorder_*.py file. `DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})` declared at module top of 5+ files. The `begin_run → register_node → create_row → create_token → begin_node_state` chain repeats verbatim in 4+ files (~200 lines saveable).

### Delete with no loss of safety

- `test_fixes.py` — disperse 2 tests; delete the file.
- `test_recorder_runs.py:20-77` — replaced by tier-1 enum-validation tests in same file.
- `test_recorder_artifacts.py:17-58` — `content_hash="abc123"` literal proves nothing.
- `test_recorder_calls.py:60-98` — superseded by `:127-165`.
- 5 of 7 `*_serialization_roundtrip.py` files — keep one parametrized + orchestrator-driven.
- `test_recorder_calls.py:319-349` — pure ORM round-trip on `None` fields.

Estimated deletion: **~3,200 lines (~22% of the chunk)** with zero loss if orchestrator-driven replacements are added.

### `test_fixes.py` triage

| Test | Verdict | Disposition |
|---|---|---|
| `test_full_plugin_discovery_flow` | Plugin discovery smoke | Move to `tests/integration/plugins/` |
| `test_dag_uses_typed_edges` | DAG enum/dataclass identity | Move to `tests/unit/core/dag/` |
| `test_error_payloads_are_structured` | `ExecutionError.to_dict()` | Move to `tests/unit/contracts/test_errors.py` |
| `test_plugin_context_recorder_can_record` | Mock-tautology | **Delete** |
| `test_edge_info_immutability` | Frozen dataclass check | Move or delete |
| `test_routing_mode_is_enum_throughout_dag` | Enum identity | Move or delete (mypy enforces) |
| `test_plugin_node_id_on_all_plugin_types` | Plugin protocol smoke | Move to `tests/integration/plugins/` |
| `test_landscape_recorder_run_lifecycle` | Duplicate of recorder_runs | **Delete** |

### Out-of-scope observations

- `tests/fixtures/landscape.py` exposes `make_recorder_with_run`, `register_test_node`, `make_factory`, `make_landscape_db` — chunk imports inconsistently.
- The `_clear_registry_for_tests` / `_snapshot_registry_for_tests` / `_restore_registry_snapshot_for_tests` triple in `declaration_contracts` appears in 6+ files — global mutable registry test-isolation hazard.
- `test_recorder_calls.py:573-700` cross-run isolation is **excellent** — promote as canonical example.
- `error_json` JSON-blob pattern bypasses any structured query primitive. `test_pass_through_violation_persists.py` is the only file that uses SQL `json_extract`.
- `test_recorder_explain.py:259+` adversarial-state test pattern should exist for every recorder query path.

---

## 3. axiom-python-engineering:python-code-reviewer

### Verdict

The audit integration test suite is largely well-structured: real DB writes, proper `tmp_path` scoping, function-scoped in-memory DB fixtures. Dominant smells: (1) **fixture shadowing** in 4 `test_recorder_row_data.py` tests — conftest fixture is created and discarded; the test runs against a local rebinding; (2) five near-identical `test_secret_like_payload_value_is_scrubbed_*` bodies; (3) three private-symbol imports (`_FlushContext`, `_cross_check_flush_output`, `_data_flow._ops`) in `test_pass_through_violation_persists.py`.

### Findings Table

| file:line | smell | severity | rationale |
|-----------|-------|----------|-----------|
| `test_recorder_row_data.py:22,101,134,172,218,265` | **Fixture shadowing** | High | Each test accepts `payload_store` (conftest `MockPayloadStore`) then immediately reassigns with `payload_store = FilesystemPayloadStore(...)`. Conftest fixture created and discarded. |
| `test_recorder_row_data.py:30` | Type-only assertion | Medium | `assert isinstance(result, RowDataResult)` followed by `assert result.state == ROW_NOT_FOUND` — isinstance redundant. |
| `test_pass_through_violation_persists.py:27` | **Private symbol import** | High | `from elspeth.engine.processor import ... _FlushContext`. Lines 134, 216, 247 call `processor._cross_check_flush_output(...)` and line 165 drills to `processor._data_flow._ops.execute_fetchone(query)`. Three layers of private coupling. |
| `test_pass_through_violation_persists.py:184` | Local import inside loop | Medium | `import json as _json` inside `for` loop body. Readability smell. |
| `test_recorder_explain.py:731` | **`pytest.raises(Exception)` without `match=`** | High | `with pytest.raises(Exception) as excinfo:` accepts any exception including `SystemExit`/`KeyboardInterrupt`. |
| `test_fixes.py:65,154` | **`ExecutionGraph()` construction bypass** | High | Manual construction in two tests. Either reclassify as unit tests or replace with `from_plugin_instances()`. |
| `test_fixes.py:208` | Hardcoded `/tmp/` path | Medium | `{"path": "/tmp/test.json", ...}` — xdist worker collision risk. |
| `test_recorder_calls.py:153–154,195–196` | Redundant type-then-value | Low | `isinstance(persisted.call_type, CallType)` then `assert persisted.call_type == CallType.LLM`. |
| `test_recorder_explain.py:403` | Stale comment as assertion substitute | Low | Comment claims production code path; no automated check enforces. |

### Production-Code-Path Bypass List

- `test_fixes.py:65` — `test_dag_uses_typed_edges`: manual `ExecutionGraph()`.
- `test_fixes.py:154` — `test_routing_mode_is_enum_throughout_dag`: same.
- `test_pass_through_violation_persists.py` — All three test methods call `processor._cross_check_flush_output()` directly.
- `test_contract_audit.py` — Uses `CSVSource` loaded manually and `MockContext` (not the orchestrated pipeline).

### Recurring Patterns

- **Fixture parameter declared but immediately overridden** — 6 times in `test_recorder_row_data.py`.
- **isinstance before equality** — 4 locations in `test_recorder_calls.py` and `test_recorder_row_data.py`.
- **Secret-scrubbing test duplicated 5 times** identically across `*_serialization_roundtrip.py` files. Only contract class name and `run_id` string differ.
- **Private three-deep attribute drilling** — `processor._data_flow._ops.execute_fetchone()` is the deepest private-chain access in the suite.
- **MockContext as structural fake** — `test_contract_audit.py:41` defines `MockContext` with only `record_validation_error`. If `PluginContext` grows a required method, tests pass but silently fail to exercise the new contract.

### Test Bodies That Should Be Deleted or Rewritten

1. `test_recorder_row_data.py` — all 6 tests in `TestGetRowDataExplicitStates`: remove the `payload_store` fixture parameter.
2. `test_fixes.py:58-77, 149-173`: move to `tests/unit/` or rewrite via `from_plugin_instances()`.
3. The 5 `test_secret_like_payload_value_is_scrubbed_*` methods: extract a shared parametrized helper.

### Pytest Hygiene Observations

- The conftest `payload_store` fixture (`tests/integration/conftest.py:55`) yields a `MockPayloadStore`, but `test_recorder_row_data.py` tests silently shadow it with `FilesystemPayloadStore`.
- `test_recorder_row_data.py:22` accepts both `tmp_path` and `payload_store`, uses `tmp_path`, ignores `payload_store`.
- No session-scoped DB fixtures with mutable state were found.
- No `autouse` fixtures with unexpected side effects were found in this subgroup.
- **No `hasattr()` calls found in this file group** — the CLAUDE.md ban is not violated here. (Notable contrast with U-CONTRACTS-1 and U-CORE-1.)

### Out-of-Scope Observations

- `test_audit_field_separation.py:132-171` patches `openai.AzureOpenAI` at the module level rather than at its import site in the LLM transform module.

**Confidence:** High. **Risk:** `/tmp/` hardcoded path is the only finding with xdist collision potential. The `pytest.raises(Exception)` is the only finding where a future crash could produce a false-positive green test. **Information Gaps:** `MockContext` divergence from `PluginContext` not exhaustively verified.

---

## 4. pr-review-toolkit:pr-test-analyzer

### Chunk verdict

This subgroup is **broad but shallow at the integration layer**. Of 32 files, only **6** drive a real `Orchestrator.run()` against `ExecutionGraph.from_plugin_instances()` end-to-end (`test_source_boundary_orchestrator.py`, `test_recorder_routing_events.py`, `test_recorder_explain.py`, `test_can_drop_rows_roundtrip.py`, `test_audit_field_separation.py`, `test_exporter_batch_queries.py`); the other 26 instantiate `RecorderFactory` and call repository methods directly — **functionally unit-style despite the directory label**. The audit legal-record spec (8-terminal closed set, recorder-was-called assertions, attributability round-trip on production path) is genuinely covered for routing/coalesce/explain, but `record_call`, content-safety, SSRF, validation-error→row linkage, composite-FK enforcement, sweep methods, and source-quarantine→QUARANTINED still have **no integration coverage**.

### Per-file scenario table

(See `i-1-findings.md` for the full table; reproduced summary here.)

The 6 keepers: `test_audit_field_separation.py`, `test_can_drop_rows_roundtrip.py`, `test_recorder_routing_events.py`, `test_recorder_explain.py` (partial), `test_source_boundary_orchestrator.py`, `test_export.py`/`test_exporter_batch_queries.py`.

### Cross-reference verification

| # | Prior-wave gap | Status | Evidence |
|---|---|---|---|
| 1 | `PluginContext.record_call()` happy + 5 crashes | **NO** | All 32 files use `factory.execution.record_call`; never `ctx.record_call`. |
| 2 | Azure content safety threshold | **NO** | No file references content_safety/unsafe/safety_filter. |
| 3 | WebScrape SSRF | **NO** | No file references SSRF/webscrape/URL-allowlist. |
| 4 | `link_validation_error_to_row` | **NO** | Zero hits across the 32 files. |
| 5 | `_REQUIRED_COMPOSITE_FOREIGN_KEYS` 11/12 untested | **NO** | No malformed-DB schema-validation tests. |
| 6 | `_validate_token_row_ownership` | **NO** | No mismatched token/row tests. |
| 7 | ADR-019 sweep methods | **NO** in this chunk | (But covered in `tests/integration/test_adr_019_*.py` — see gap-analyst.) |
| 8 | Attributability round-trip Source→Transform→Sink + `explain()` | **PARTIAL** | Only `test_recorder_explain.py::test_union_merge_surfaces_field_provenance_via_explain` (line 522) and `test_recorder_routing_events.py::test_explain_recovers_routing_intent_for_both_variants` (line 538) round-trip via Orchestrator+explain. No simple linear baseline. |
| 9 | 8-terminal closed-set coverage | **PARTIAL** (5 of 8 hit on production path) | SUCCESS, FAILURE, FAILURE-source-boundary, FILTER_DROPPED, COALESCED. Missing on production path: COMPLETED-linear, FORKED, EXPANDED, CONSUMED_IN_BATCH, QUARANTINED. |
| 10 | Recorder-was-called assertions on contract happy paths | **PARTIAL** | `test_audit_field_separation.py` checks `success_reason_json` populated. No happy-path sink contract test asserts `recorder.record_*` invoked. |

### Critical scenario gaps (chunk-wide)

1. Source quarantine → QUARANTINED terminal under real DAG (severity 9)
2. `PluginContext.record_call` integration through running orchestrator (severity 9)
3. Linear Source→Transform→Sink happy-path attributability baseline (severity 8)
4. FORKED, EXPANDED, CONSUMED_IN_BATCH terminals on production path (severity 8)
5. Composite-FK enforcement against malformed DB (severity 8)
6. `link_validation_error_to_row` from a real source (severity 8)
7. ADR-019 sweep on orphan parent outcome (severity 7)
8. Tier-1 read-side guards (severity 7)
9. Hash-survives-payload-deletion in real pipeline (severity 6)
10. Resume after crash audit consistency (severity 6)

### Low-effort / pointless tests

- `test_fixes.py:25, :58, :79, :105, :133, :149, :175, :213` — 8 tests, none of which are audit tests.
- `test_recorder_artifacts.py:228, :257, :291` — pure CRUD round-trips.
- `test_recorder_runs.py:81, :115` — duplicates of unmarked enum versions.
- `test_recorder_calls.py:351` — `assert created_at is not None`; doesn't pin UTC, monotonicity, or canonical timestamp format.
- `test_recorder_calls.py:319, :333` — round-trips of None into None.
- `test_recorder_nodes.py:20, :39, :67` — asserts nodes round-trip.

`test_fixes.py` should be **deleted entirely** — every test is a "Task N landed" anchor. Per CLAUDE.md "no legacy code", historical task verification belongs in commit messages, not tests.

### Out-of-scope observations

- Helpers from `tests.fixtures.*` and `from elspeth.testing import make_pipeline_row` used inconsistently.
- `test_recorder_calls.py` line 1 has stale path comment.
- `test_export.py` invokes `cli.app` via `CliRunner` — only CLI-driven audit integration test in chunk; assertions stop at "record_type set is non-empty"; could anchor far more.

---

## 5. ordis-quality-engineering:coverage-gap-analyst

### SUT Footprint Summary

(See findings synthesis. Chunk's strongest files: `test_tier1_integrity.py`, `test_recorder_tokens.py`, `test_recorder_explain.py`. Chunk's weakest: `test_fixes.py`, repository-direct `test_recorder_*.py` files.)

### Cross-Reference Verdicts

| Prior-Wave Gap | Status | Evidence |
|---|---|---|
| `PluginContext.record_call()` — 5 crash branches | **Open (partial downgrade).** All 32 files call `landscape_factory.execution.record_call()` directly, bypassing the PluginContext wrapper. None exercise the XOR/landscape-None/state-mismatch/token-mismatch guard paths. | `test_recorder_calls.py` — all calls via `landscape_factory.execution`. |
| Azure content safety threshold | Out of scope (plugin-tier). | No relevant SUT in `core/landscape/`. |
| WebScrape SSRF / `_final_response_ip` | Out of scope (plugin-tier). | No relevant SUT in `core/landscape/`. |
| `DataFlowRepository.link_validation_error_to_row` | **Open (critical).** Zero hits anywhere in `tests/`. Four crash branches (cross-run row, nonexistent error, cross-run error, conflicting prior link) are completely untested at any layer. | `grep -rn "link_validation_error_to_row" tests/` → empty. |
| `_REQUIRED_COMPOSITE_FOREIGN_KEYS` (12 entries, 1 covered) | **Partially covered at unit layer only.** Mock inspector for `transform_errors` only. No integration test exercises a real DB physically lacking one of the other 11 composite FKs. | `test_validate_schema_rejects_stale_single_column_foreign_keys_for_run_scoped_error_tables` (unit, mock). |
| `DataFlowRepository._validate_token_row_ownership` | **Open.** Zero test hits anywhere in the test suite. Called from three write sites; cross-run mismatch branch never asserted. | `grep -rn "_validate_token_row_ownership" tests/` → empty. |
| ADR-019 `sweep_deferred_invariants_or_crash`, `find_orphaned_*` | **Covered outside this cluster.** `tests/integration/test_adr_019_cross_table_invariants.py` (lines 293, 322, 347, 378) and `test_adr_019_sweep_durability.py`. | Not a gap; note location. |

### New Critical Gaps (Integration-Layer Only)

| File:Function | Gap | Why Critical |
|---|---|---|
| `data_flow_repository.py:link_validation_error_to_row` | All 4 crash branches untested | Tier-1 — wrong linkage is evidence tampering |
| `data_flow_repository.py:_validate_token_row_ownership` | Cross-run ownership mismatch branch never exercised | A token from run A could be silently written into run B's outcome table |
| `contracts/plugin_context.py:record_call` + `Orchestrator` | PluginContext crash branches never exercised through real DAG run | A broken orchestrator failing to inject landscape would silently record nothing |
| `Orchestrator` + failing transform | No test runs `from_plugin_instances → Orchestrator.run` where a transform `process()` raises | "Silent wrong result is worse than a crash" — no integration test confirms this contract |

### High-Risk Gaps

- **CONSUMED_IN_BATCH at audit-recorder layer** (under real orchestration) — covered at unit layer in `test_batch_token_identity.py` and at engine integration in `test_t18_characterization.py`, but not asserted at the audit-recorder layer within this cluster.
- **`_REQUIRED_COMPOSITE_FOREIGN_KEYS` entries 2–12** — no real-DB integration test populates a real schema lacking entries 2–12.
- **`lineage.py:explain` + real full Source→Transform→Sink pipeline** — fancy fork/coalesce explain coverage but no degenerate-DAG attributability baseline.

### Quick Wins

- `link_validation_error_to_row` — pure DB write; ~4 focused tests, ~50 lines total.
- `_validate_token_row_ownership` — same pattern; closes cross-run write-guard gap.

### Notable Strengths

- `test_tier1_integrity.py` — Tier-1 trust model well-defended at DB layer.
- `test_pass_through_violation_persists.py` — "no row left behind" guarantee has real integration guard.
- `test_recorder_explain.py` union-merge tests — three real Orchestrator runs covering last_wins, fail, and first_wins with provenance asserted on both raw SQL column and `explain()` API.
- `test_not_null_constraints.py` and `test_sqlcipher_pipeline.py` — DB-level enforcement and encryption-at-rest verified end-to-end.

### Confidence / Risk / Information Gaps / Caveats

**High confidence** for `link_validation_error_to_row`, `_validate_token_row_ownership`, ADR-019 sweep (covered elsewhere). **Medium confidence** for PluginContext integration gap. **Information Gaps:** `tests/integration/plugins/` subtree not exhaustively searched. **Caveats:** CONSUMED_IN_BATCH gap is specific to the audit cluster — coverage exists in other clusters.
