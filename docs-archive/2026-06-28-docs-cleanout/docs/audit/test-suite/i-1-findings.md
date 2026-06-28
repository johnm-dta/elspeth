# I-1 — Synthesised Findings

**Scope:** 32 files in `tests/integration/audit/`.
**Method:** 5 specialist agents in parallel + cross-reference task against prior-wave findings.
**Date:** 2026-05-06.

## Verdict

**Health: Concerning, leaning Poor at the structural level.** The chunk *appears* to provide audit integration coverage but **only ~6 of 32 files actually do** (`test_audit_field_separation.py`, `test_can_drop_rows_roundtrip.py`, `test_recorder_explain.py`, `test_recorder_routing_events.py`, `test_source_boundary_orchestrator.py`, `test_export.py`/`test_exporter_batch_queries.py`). The remaining 26 are repository tests in an integration directory — they verify `RecorderFactory` write/read round-trips but never exercise the production code path that production plugins use.

This produces a **false-confidence trap**: passing tests in `tests/integration/audit/` are read by reviewers as "audit DB is verified end-to-end," when in reality only the repository layer is covered.

Per-lens verdicts: Mixed (anti-patterns), **Pervasive theatre** (qa-analyst), Sound-but-bypass-pattern (Python smells), Broad-but-shallow (scenario coverage), 6/8 prior-wave gaps still open at integration layer (gap analyst).

## Convergent findings (≥2 agents agree)

### CONV-1 — `test_fixes.py` is a regression dumping ground — 5/5 agents

Header cites "Tasks 1-7"; contents are non-audit tests:
- `test_full_plugin_discovery_flow` (`:25-56`) — assignment tautology, no Landscape write
- `test_dag_uses_typed_edges` (`:58-77`) — `ExecutionGraph()` constructed manually (production-path bypass)
- `test_edge_info_immutability` (`:133-147`) — tests Python's `frozen=True`
- `test_routing_mode_is_enum_throughout_dag` (`:149-173`) — asserts type of values the test inserted
- `test_plugin_node_id_on_all_plugin_types` (`:175-211`) — three-way assignment tautology + hardcoded `/tmp/test.json` (xdist collision risk)
- `test_plugin_context_recorder_can_record` (`:105-131`) — mock-tautology
- `test_landscape_recorder_run_lifecycle` (`:213-241`) — duplicate of `test_recorder_runs.py`

**Recommendation (unanimous):** disperse 2 tests with any value to proper homes; delete the file.

### CONV-2 — 26 of 32 files are unit tests masquerading as integration tests — 4/5 agents

Per CLAUDE.md: "Integration tests MUST use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()`." Only the 6 keepers above use it. The remaining 26 (bulk of `test_recorder_*.py` and the 9-file roundtrip cluster) call `factory.execution.record_call(...)` or `factory.data_flow.create_token(...)` directly.

**Recommendation:** either (a) move the 26 files to `tests/unit/core/landscape/repository_integration/` to truthfully label them, or (b) rewrite each through `Orchestrator.run()`. Option (a) is mechanical; option (b) is the right thing.

### CONV-3 — 9-file `*_serialization_roundtrip.py` copy-paste cluster — 3/5 agents

Files: `test_declaration_contract_landscape_serialization_roundtrip.py`, `test_declared_output_fields_serialization_roundtrip.py`, `test_declared_required_fields_serialization_roundtrip.py`, `test_source_guaranteed_fields_serialization_roundtrip.py`, `test_sink_required_fields_serialization_roundtrip.py`, `test_schema_config_mode_serialization_roundtrip.py`, `test_can_drop_rows_roundtrip.py` (outlier — uses Orchestrator), plus 2 sibling tests.

~85% identical: `_setup_landscape()`, `_record_failure()`, `setup_method`/`teardown_method` snapshot/restore, the `try/except violation: error = ExecutionError(... violation.to_audit_dict()); else: assert False` pattern, `assert "sk-abcdef" not in json.dumps(context)` secret-redaction tail (5 verbatim copies), `_RoundTripPayload` TypedDict.

**Recommendation:** consolidate to one parametrized file. ~2,400 → ~400 lines. The 5-fold duplicated secret-redaction test moves to a single `test_secret_redaction.py`.

### CONV-4 — Hash-without-binding theatre persists at integration layer — 2/5 agents

- `test_recorder_artifacts.py:17-58, 113-178` — `content_hash="abc123"` literal, never derived from artifact bytes
- `test_recorder_calls.py:60-98, 351-362` — `assert call.request_hash is not None` (presence-only)
- `test_recorder_runs.py:20-77` — pure `factory.x → assert x.field == value-just-passed`

### CONV-5 — Stale path comments in 5 files — 2/5 agents

`test_recorder_calls.py:1`, `test_recorder_contracts.py:1`, `test_recorder_nodes.py:1`, `test_recorder_row_data.py:1`, `test_recorder_runs.py:1` open with `# tests/core/landscape/...` (pre-migration path).

### CONV-6 — Fixture and import inconsistency — 2/5 agents

- `DYNAMIC_SCHEMA = SchemaConfig.from_dict(...)` declared at module top of 5+ files; should move to `tests/fixtures/landscape.py`
- `begin_run → register_node → create_row → create_token → begin_node_state` chain repeats verbatim in 4+ files (~200 lines saveable)
- `test_recorder_explain.py:27` imports from another test file
- `test_recorder_row_data.py` 6 tests accept `payload_store` then immediately rebind it (fixture shadowing)

## Single-lens findings worth surfacing

### SOLO-1 — `test_error_persistence.py:161-175, 246-260` raw SQL token inserts (Critical)

Two production-path bypasses in a file that purports to test error persistence. Raw `conn.execute(tokens_table.insert().values(token_id="token-123", ...))` with comment "to match test expectations." Bypasses `factory.data_flow.create_token()`.

### SOLO-2 — `test_pass_through_violation_persists.py` three-deep private drilling (High)

`processor._cross_check_flush_output(...)` (lines 134, 216, 247) and `processor._data_flow._ops.execute_fetchone(query)` (line 165). Plus `_FlushContext` imported by name. Refactoring internal processor structure silently breaks tests.

### SOLO-3 — `test_recorder_explain.py:731` `pytest.raises(Exception)` without `match=` (High)

Accepts any exception including `SystemExit`/`KeyboardInterrupt`. Subsequent chain-walk to `CoalesceCollisionError` doesn't replace the `match=` guard.

### SOLO-4 — `test_contract_audit.py:621-635` defensive `if restored:` guards on Tier-1 data (Minor)

CLAUDE.md: Tier-1 data must crash on anomaly. `if restored1: ...` silently skips the assertion if `get_run_contract` returns `None` after storage. Should be unconditional.

### SOLO-5 — `MockContext` in `test_contract_audit.py` structural fake (Medium)

Implements `PluginContext` with only `record_validation_error`. If `PluginContext` grows a required method, these tests pass but silently fail to exercise the new contract.

## Cross-reference: prior-wave gaps × integration layer

| # | Prior-wave gap | Filed | Status at integration layer |
|---|---|---|---|
| 1 | `PluginContext.record_call()` — happy + 5 crash branches | `elspeth-f92ba560ad` | **Still open.** All 32 files use `factory.execution.record_call` (repository), never `ctx.record_call` (the wrapper plugins use). |
| 2 | Azure content safety threshold rejection | `elspeth-8d5558dc25` | Not in scope (plugin-tier). |
| 3 | WebScrape SSRF boundary | `elspeth-7b7fe68836` | Not in scope (plugin-tier). |
| 4 | `link_validation_error_to_row` | `elspeth-297dafdf47` | **Doubly confirmed open.** Zero hits at integration AND unit layer. |
| 5 | `_REQUIRED_COMPOSITE_FOREIGN_KEYS` 11/12 untested | `elspeth-499100db05` | **Still open at integration.** No real-DB schema-validation tests with malformed FK shapes. |
| 6 | `_validate_token_row_ownership` | `elspeth-82c7c028a8` | **Doubly confirmed open.** Zero hits at any layer. |
| 7 | ADR-019 sweep methods | `elspeth-f6f50e9394` | **DOWNGRADE SCOPE.** Covered in `tests/integration/test_adr_019_cross_table_invariants.py` (lines 293, 322, 347, 378) and `test_adr_019_sweep_durability.py`. Original issue should be re-scoped from "no tests" to "no *unit* tests." |

## New gaps surfaced at the integration layer

| Gap | Severity |
|---|---|
| Source quarantine → QUARANTINED terminal under real DAG | Critical |
| `PluginContext.record_call` integration through running orchestrator | Critical |
| Linear Source→Transform→Sink happy-path attributability baseline | High |
| FORKED, EXPANDED, CONSUMED_IN_BATCH terminals on production path | High |
| Composite-FK enforcement against malformed real DB | High |
| Crash propagation: transform raises → run crashes (no silent swallow) | High |
| Tier-1 read-side guards | High |
| Resume after crash audit consistency | Medium |

## Top deletion candidates (consensus)

| # | Target | Lines | Confidence |
|---|---|---|---|
| 1 | `test_fixes.py` (entire file, after dispersing 2 keepers) | ~241 | High |
| 2 | 5 of 7 `*_serialization_roundtrip.py` files (consolidate to 1 parametrized + 1 secret-redaction test) | ~2,000 net | High |
| 3 | `test_recorder_runs.py:20-77` (4 round-trip tests) | ~57 | High |
| 4 | `test_recorder_artifacts.py:17-58, 113-178` | ~120 | High |
| 5 | `test_recorder_calls.py:319-349` | ~30 | High |
| 6 | `test_recorder_calls.py:60-98` (presence-only hash) | ~38 | Medium |
| 7 | `test_recorder_explain.py:36+` lifted to fixture | (refactor) | Medium |
| 8 | Stale `# tests/core/landscape/...` headers in 5 files | 5 lines | High |
| 9 | `test_error_persistence.py:161-175, 246-260` raw SQL → `factory.data_flow.create_token` | (rewrite) | High |
| 10 | `test_recorder_calls.py:153-154, 195-196` redundant `isinstance + ==` | ~4 lines | Medium |

**Total deletable + consolidatable: ~2,500 lines / ~50-70 test bodies.**

## Top "add immediately" candidates

1. **`PluginContext.record_call` through real orchestrator** — close `elspeth-f92ba560ad` at integration layer.
2. **Source quarantine → QUARANTINED + linked validation_error** — closes both new chunk gap and `elspeth-297dafdf47` at integration.
3. **Composite-FK against real malformed DB** — close `elspeth-499100db05` at integration.
4. **Cross-run `_validate_token_row_ownership`** integration test — close `elspeth-82c7c028a8`.
5. **Linear Source→Transform→Sink + `explain()` attributability baseline**.
6. **Terminal-state production-path coverage** — at least one test per terminal under `Orchestrator.run()`.
7. **Crash-propagation test** — transform raises → audit trail shows FAILED token with non-null error hash.

## Notable strengths (preserve and use as templates)

- `test_audit_field_separation.py` — exemplary: real `Orchestrator`, mocks only Tier-3 Azure boundary.
- `test_recorder_routing_events.py` — gold standard for orchestrator-driven routing.
- `test_recorder_explain.py` union-merge tests — thorough attributability under merge.
- `test_tier1_integrity.py` — comprehensive write-side Tier-1 crash coverage.
- `test_pass_through_violation_persists.py` — explicit docstring justification for SQL-level `json_extract` queries.
- `test_recorder_calls.py:573-700` cross-run isolation — canonical good-integration-test example.

## Filed filigree issues

### Remediation status

- 2026-05-20: `elspeth-4a013f9833` resolved. The salvageable `ExecutionGraph` typed-edge, `ExecutionError.to_dict()`, and built-in plugin `node_id` assertions were moved to unit/plugin contract homes, and `tests/integration/audit/test_fixes.py` was deleted.
- 2026-05-20: `elspeth-ae9f541775` resolved via the recommended Option A. The 25 remaining repository-level audit persistence tests were moved to `tests/unit/core/landscape/repository_integration/`, `tests/integration/audit/` now contains only the orchestrator/production-path audit tests, and `tests/unit/test_audit_integration_directory_discipline.py` mechanically guards the split.

| ID | Title | Type | Priority | Status |
|---|---|---|---|---|
| `elspeth-ae9f541775` | tests/integration/audit/ — 26 of 32 files bypass production code paths (false-confidence integration coverage) | epic | P1 | Closed 2026-05-20 |
| `elspeth-4a013f9833` | tests/integration/audit/test_fixes.py — regression-dump file unanimously flagged for deletion/dispersal | task | P2 | Closed 2026-05-20 |

### Existing issue updated

- `elspeth-f6f50e9394` (ADR-019 sweep methods) — comment added explaining scope revision: integration coverage exists in `tests/integration/test_adr_019_*.py`. Original issue should be re-scoped from "no tests" to "no *unit* tests"; severity downgrade recommended from P0 to P2/P3.

### Issues NOT filed (covered by existing items)

- `PluginContext.record_call` integration through running orchestrator — already tracked under `elspeth-f92ba560ad`. The I-1 finding extends scope from unit-only to "also missing at integration layer"; no new issue needed.
- `link_validation_error_to_row` at integration layer — already tracked under `elspeth-297dafdf47`. Doubly confirmed open.
- `_REQUIRED_COMPOSITE_FOREIGN_KEYS` against real malformed DB — already tracked under `elspeth-499100db05`.
- `_validate_token_row_ownership` at integration layer — already tracked under `elspeth-82c7c028a8`. Doubly confirmed open.

The cross-reference task validated 4 of 7 prior issues and surfaced the structural epic + the regression-dump task as the genuinely new findings worth filing.

## Out-of-scope observations

1. `test_recorder_explain.py:27` cross-test-file imports — fragile; helper should live in `tests/fixtures/`.
2. `test_audit_field_separation.py:132-171` patches `openai.AzureOpenAI` at module level; if LLMTransform changes import path, patch becomes stale silently.
3. `_clear_registry_for_tests` / `_snapshot_registry_for_tests` / `_restore_registry_snapshot_for_tests` triple appears in 6+ files — global mutable registry test-isolation hazard worth a separate issue.
4. `test_export.py:230-460` signed-export hash determinism may be testing rfc8785 (canonical) rather than the export pipeline. Verify the `signing_key` path.
5. Helpers from `tests.fixtures.*` and `from elspeth.testing import make_pipeline_row` are used inconsistently across files — fixture-layer audit pass warranted.
