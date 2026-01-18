# Plugin Refactor Progress Tracker

> **Created:** 2026-01-17
> **Source:** work-packages.md, gap-analysis.md
> **Contract:** plugin-protocol.md v1.1
> **Total Effort:** ~70 hours

---

## Quick Status

| WP | Name | Status | Effort | Dependencies | Unlocks |
|----|------|--------|--------|--------------|---------|
| WP-01 | Protocol & Base Class Alignment | 🟢 Complete | 2h | None | WP-03 |
| WP-02 | Gate Plugin Deletion | 🟢 Complete | 1h | None | WP-09 |
| WP-03 | Sink Implementation Rewrite | 🟢 Complete | 4h | WP-01 | WP-04, WP-13 |
| WP-04 | Delete SinkAdapter & SinkLike | 🟢 Complete | 2h | WP-03 | WP-04a, WP-13 |
| WP-04a | Delete *Like Protocol Duplications | 🟢 Complete | 1.5h | WP-04 | — |
| WP-05 | Audit Schema Enhancement | 🟢 Complete | 2h | None | WP-06 |
| WP-06 | Aggregation Triggers | 🟢 Complete | 6h | WP-05 | WP-14 |
| WP-07 | Fork Work Queue | 🟢 Complete | 8h | None | WP-08, WP-10 |
| WP-08 | Coalesce Executor | 🟢 Complete | 8h | WP-07 | WP-14 |
| WP-09 | Engine-Level Gates | 🟢 Complete | 10h | (after WP-02) | WP-14 |
| WP-10 | Quarantine Implementation | 🟢 Complete | 4h | WP-07 | WP-14 |
| WP-11 | Orphaned Code Cleanup | 🟢 Complete | 2h | None | — |
| WP-11.99 | Config-Driven Plugin Schemas | 🟢 Complete | 4-6h | None | WP-12 |
| WP-12 | Utility Consolidation | 🟢 Complete | 0.5h | WP-11.99 | — |
| WP-13 | Sink Test Rewrites | 🟢 Complete | 4h | WP-03, WP-04 | — |
| WP-14 | Engine Test Rewrites | 🔴 Not Started | 16h | WP-06,07,08,09,10 | — |
| WP-15 | RetryManager Integration | 🟢 Complete | 4h | None | — |

**Legend:** 🔴 Not Started | 🟡 In Progress | 🟢 Complete | ⏸️ Blocked

---

## Dependency Graph

```
WP-01 ──┬──► WP-03 ──► WP-04 ──┬──► WP-04a
        │                      └──► WP-13
WP-02   │   (independent)

WP-05 ──┴──► WP-06
                        ╲
WP-07 ──┬──► WP-08 ──────┬──► WP-14
        └──► WP-10 ──────╱

WP-09 ──────────────────╱

WP-11       (independent)
WP-11.99 ──► WP-12  (config-driven schemas unlock simplified utility consolidation)
WP-15       (independent - RetryManager integration)
```

---

## Sprint Allocation

> **IMPORTANT:** WP-02 and WP-09 MUST be back-to-back (no gate gap)

### Sprint 1: Foundation
- [x] WP-01: Protocol & Base Class Alignment
- [x] WP-05: Audit Schema Enhancement ✅ Complete (2026-01-18)
- [x] WP-11: Orphaned Code Cleanup ✅ Complete (2026-01-18)

### Sprint 2: Sink Contract & Interface Cleanup
- [x] WP-03: Sink Implementation Rewrite
- [ ] WP-04: Delete SinkAdapter & SinkLike
- [ ] WP-04a: Delete *Like Protocol Duplications (TransformLike, GateLike, AggregationLike)
- [x] WP-12: Utility Consolidation ✅ Complete (2026-01-18)
- [x] WP-13: Sink Test Rewrites ✅ Complete (done with WP-03)

### Sprint 3: DAG & Aggregation
- [x] WP-06: Aggregation Triggers ✅ Complete (2026-01-18)
- [x] WP-07: Fork Work Queue ✅ Complete (2026-01-18)
- [x] WP-10: Quarantine Implementation ✅ Complete (2026-01-18)

### Sprint 4: Gates & Coalesce
- [x] WP-02: Gate Plugin Deletion ✅ Complete (2026-01-18)
- [x] WP-09: Engine-Level Gates ✅ Complete (2026-01-18)
- [x] WP-08: Coalesce Executor ✅ Complete (2026-01-18)

### Sprint 5: Verification & Integration
- [ ] WP-14: Engine Test Rewrites (split into WP-14a/b/c/d/e)
- [x] WP-15: RetryManager Integration ✅ Complete (2026-01-18)
- [ ] Final integration testing

---

## Detailed Work Package Tracking

---

### WP-01: Protocol & Base Class Alignment

**Status:** 🟢 Complete (2026-01-17)
**Plan:** [2026-01-17-wp01-protocol-alignment.md](./2026-01-17-wp01-protocol-alignment.md)
**Goal:** Align SourceProtocol and SinkProtocol with contract v1.1

**Files:**
- `src/elspeth/plugins/protocols.py`
- `src/elspeth/plugins/base.py`
- `tests/plugins/test_protocols.py`
- `tests/plugins/test_base.py`

#### Task 1: Add determinism and plugin_version to SourceProtocol
- [x] Write failing tests (`test_source_has_determinism_attribute`, `test_source_has_version_attribute`, `test_source_implementation_with_metadata`)
- [x] Run tests to verify they fail
- [x] Add `determinism: Determinism` to SourceProtocol (protocols.py:52-54)
- [x] Add `plugin_version: str` to SourceProtocol (protocols.py:52-54)
- [x] Run tests to verify they pass
- [x] Commit: `329a121` - `feat(protocols): add determinism and plugin_version to SourceProtocol`

#### Task 2: Add determinism and plugin_version to BaseSource
- [x] Write failing tests (`test_base_source_has_metadata_attributes`, `test_subclass_can_override_metadata`)
- [x] Run tests to verify they fail
- [x] Add `determinism: Determinism = Determinism.IO_READ` to BaseSource (base.py:321-324)
- [x] Add `plugin_version: str = "0.0.0"` to BaseSource (base.py:321-324)
- [x] Run tests to verify they pass
- [x] Commit: `bba40c5` - `feat(base): add determinism and plugin_version to BaseSource`

#### Task 3: Update SinkProtocol.write() signature to batch mode
- [x] Write failing tests (`test_sink_batch_write_signature`, `test_batch_sink_implementation`)
- [x] Run tests to verify they fail
- [x] Update SinkProtocol.write(): `write(row: dict) -> None` → `write(rows: list[dict]) -> ArtifactDescriptor`
- [x] Add ArtifactDescriptor import to protocols.py TYPE_CHECKING block
- [x] Run new tests to verify they pass
- [x] Update old `test_sink_implementation` to use new signature
- [x] Run all SinkProtocol tests
- [x] Commit: `fd0b29a` - `feat(protocols): update SinkProtocol.write() to batch mode`

#### Task 4: Update BaseSink.write() signature to batch mode
- [x] Write failing tests (`test_base_sink_batch_write_signature`, `test_base_sink_batch_implementation`)
- [x] Run tests to verify they fail
- [x] Update BaseSink.write(): `write(row: dict) -> None` → `write(rows: list[dict]) -> ArtifactDescriptor`
- [x] Change BaseSink.determinism default: `DETERMINISTIC` → `IO_WRITE`
- [x] Add ArtifactDescriptor import to base.py
- [x] Run new tests to verify they pass
- [x] Update old `test_base_sink_implementation` to use new signature
- [x] Run all BaseSink tests
- [x] Commit: `761d757` - `feat(base): update BaseSink.write() to batch mode`

#### Task 5: Verify type checking passes
- [x] Run `mypy src/elspeth/plugins/protocols.py src/elspeth/plugins/base.py --strict` ✅ Clean
- [x] Run `pytest tests/plugins/test_protocols.py tests/plugins/test_base.py -v` ✅ 32 tests pass
- [x] Document expected failures in sink implementations (WP-03 will fix)
- [x] Verification complete (no new commit - verification only)

#### Verification Checklist
- [x] `SourceProtocol` has `determinism: Determinism` attribute
- [x] `SourceProtocol` has `plugin_version: str` attribute
- [x] `BaseSource` has `determinism = Determinism.IO_READ` default
- [x] `BaseSource` has `plugin_version = "0.0.0"` default
- [x] `SinkProtocol.write()` signature is `write(rows: list[dict], ctx) -> ArtifactDescriptor`
- [x] `BaseSink.write()` signature is `write(rows: list[dict], ctx) -> ArtifactDescriptor`
- [x] `BaseSink` has `determinism = Determinism.IO_WRITE` default
- [x] `mypy --strict` passes on protocols.py and base.py
- [x] All plugin tests pass (32/32)

#### Known Issues for WP-03
6 mypy errors in sink implementations (expected - old signature):
- `csv_sink.py:53` - write() signature mismatch
- `json_sink.py:59` - write() signature mismatch
- `database_sink.py:80` - write() signature mismatch

#### Commits
```
329a121 feat(protocols): add determinism and plugin_version to SourceProtocol
bba40c5 feat(base): add determinism and plugin_version to BaseSource
fd0b29a feat(protocols): update SinkProtocol.write() to batch mode
761d757 feat(base): update BaseSink.write() to batch mode
```

---

### WP-02: Gate Plugin Deletion

**Status:** 🟢 Complete (2026-01-18)
**Plan:** [2026-01-18-wp02-gate-plugin-deletion.md](./2026-01-18-wp02-gate-plugin-deletion.md)
**Goal:** Complete removal of plugin-based gates (engine gates come in WP-09)

#### Files DELETED (9 files, ~1,350 lines)
- [x] `src/elspeth/plugins/gates/filter_gate.py` (249 lines)
- [x] `src/elspeth/plugins/gates/field_match_gate.py` (193 lines)
- [x] `src/elspeth/plugins/gates/threshold_gate.py` (144 lines)
- [x] `src/elspeth/plugins/gates/hookimpl.py` (22 lines)
- [x] `src/elspeth/plugins/gates/__init__.py` (11 lines)
- [x] `tests/plugins/gates/test_filter_gate.py` (276 lines)
- [x] `tests/plugins/gates/test_field_match_gate.py` (230 lines)
- [x] `tests/plugins/gates/test_threshold_gate.py` (221 lines)
- [x] `tests/plugins/gates/__init__.py` (1 line)

#### Files MODIFIED (9 files)
- [x] `src/elspeth/cli.py` - Remove gate imports and GATE_PLUGINS registry
- [x] `src/elspeth/plugins/manager.py` - Remove builtin_gates import and registration
- [x] `tests/plugins/test_base.py` - Remove TestBaseGate class
- [x] `tests/plugins/test_protocols.py` - Remove TestGateProtocol class
- [x] `tests/plugins/test_integration.py` - Remove gate from workflow test
- [x] `tests/engine/test_plugin_detection.py` - Remove gate import tests
- [x] `tests/plugins/test_hookimpl_registration.py` - Remove gate discovery test
- [x] `tests/integration/test_audit_integration_fixes.py` - Fix gate assertion
- [x] `tests/cli/test_run_with_row_plugins.py` - Remove gate CLI tests (plan gap fix)

#### Preserved for WP-09
- `GateProtocol` in protocols.py
- `BaseGate` in base.py
- `GateResult`, `RoutingAction` in contracts/results.py
- `PluginManager.get_gates()`, `get_gate_by_name()` infrastructure

#### Commits (11 total)
```
c9924af refactor(gates): delete plugin-based gate implementations
76b21c7 test(gates): delete plugin-based gate tests
9a03f86 refactor(cli): remove gate plugin references
01f0a96 refactor(manager): remove gate plugin registration
61c7879 test(base): remove BaseGate test class
46d7598 test(protocols): remove GateProtocol conformance test
647d41e test(integration): remove gate from plugin workflow test
02eacc4 test(detection): remove gate plugin import tests
3155ce3 test(plugins): remove gate discovery test (WP-02 Task 9)
c231f41 test(integration): update gate assertion for WP-02 (Task 10)
2496151 test(cli): remove gate plugin tests (WP-02 verification fix)
```

#### Verification (2026-01-18)
- [x] `grep -r "FilterGate\|FieldMatchGate\|ThresholdGate" src/` returns nothing
- [x] No imports of deleted gate plugins anywhere
- [x] mypy passes (84 files, no issues)
- [x] pytest passes (1203 tests, 2 pre-existing failures unrelated to WP-02)

---

### WP-03: Sink Implementation Rewrite

**Status:** 🟢 Complete
**Goal:** All sinks conform to batch signature with ArtifactDescriptor return
**Blocked by:** WP-01 ✅

#### Sinks to Rewrite
- [x] `src/elspeth/plugins/sinks/csv_sink.py`
  - [x] Change `write(row) -> None` to `write(rows) -> ArtifactDescriptor`
  - [x] Implement SHA-256 content hashing of written file
  - [x] Add `determinism = Determinism.IO_WRITE`
  - [x] Add `plugin_version = "1.0.0"`
  - [x] Add `on_start()` and `on_complete()` lifecycle hooks

- [x] `src/elspeth/plugins/sinks/json_sink.py`
  - [x] Change `write(row) -> None` to `write(rows) -> ArtifactDescriptor`
  - [x] Implement SHA-256 content hashing of written file
  - [x] Add `determinism = Determinism.IO_WRITE`
  - [x] Add `plugin_version = "1.0.0"`
  - [x] Add `on_start()` and `on_complete()` lifecycle hooks

- [x] `src/elspeth/plugins/sinks/database_sink.py`
  - [x] Change `write(row) -> None` to `write(rows) -> ArtifactDescriptor`
  - [x] Implement SHA-256 of canonical JSON payload before INSERT
  - [x] Add `determinism = Determinism.IO_WRITE`
  - [x] Add `plugin_version = "1.0.0"`
  - [x] Add `on_start()` and `on_complete()` lifecycle hooks

#### Commits
```
1a4f414 feat(csv-sink): implement batch write with ArtifactDescriptor
57e2b65 feat(json-sink): implement batch write with ArtifactDescriptor
58685dd feat(database-sink): implement batch write with ArtifactDescriptor
5b309ba feat(sinks): add explicit lifecycle hooks to all sinks
```

#### Verification (2026-01-17)
- [x] All sinks return ArtifactDescriptor
- [x] content_hash is non-empty SHA-256
- [x] size_bytes > 0 for non-empty writes
- [x] Mypy --strict passes on all sink files
- [x] 41 sink tests pass
- [x] No per-row write(row) calls remain in tests
- [x] All sinks have `on_start()` and `on_complete()` lifecycle hooks
- [x] All sinks have `determinism == Determinism.IO_WRITE` (inherited from BaseSink)

---

### WP-04: Delete SinkAdapter & SinkLike

**Status:** 🟢 Complete
**Plan:** [2026-01-17-wp04-sink-adapter-update.md](./2026-01-17-wp04-sink-adapter-update.md)
**Goal:** Remove adapter layer - sinks now implement batch interface directly
**Blocked by:** WP-03 ✅
**Completed:** 2026-01-17 (commit f08c19a)

**Rationale:** WP-03 made sinks batch-aware with ArtifactDescriptor returns. SinkAdapter and SinkLike are now redundant indirection layers.

#### Tasks
- [x] Task 1: Delete `adapters.py` and `test_adapters.py`
- [x] Task 2: Delete `SinkLike` from `executors.py`
- [x] Task 3: Update `orchestrator.py` to use `SinkProtocol`
- [x] Task 4: Update CLI to use sinks directly
- [x] Task 5: Remove `SinkAdapter` from `engine/__init__.py` exports
- [x] Task 6: Run full verification

#### Verification
- [x] `adapters.py` deleted
- [x] `test_adapters.py` deleted
- [x] No `SinkLike` anywhere in codebase
- [x] No `SinkAdapter` anywhere in codebase
- [x] CLI creates sinks directly (no wrapper)
- [x] Orchestrator uses `SinkProtocol` type hints
- [x] All tests pass (167 engine tests)

---

### WP-04a: Delete *Like Protocol Duplications

**Status:** 🟢 Complete
**Plan:** [2026-01-17-wp04a-delete-like-protocols.md](./2026-01-17-wp04a-delete-like-protocols.md)
**Goal:** Delete AggregationLike protocol, move batch state to executor, rename TransformLike alias
**Completed:** 2026-01-17 (commits 6ff0d49, 2b82bf2, f9153d0, 3657d92)

#### Tasks
- [x] TransformLike protocol deleted (commit f08c19a)
- [x] GateLike protocol deleted (commit f08c19a)
- [x] Task 1: Refactor AggregationExecutor to store batch state internally (Option C)
- [x] Task 2: Delete AggregationLike protocol
- [x] Task 3: Rename TransformLike union alias to RowPlugin
- [x] Task 4: Update tests (use executor.get_batch_id() instead of aggregation._batch_id)
- [x] Task 5: Final verification

**Files Changed:**
- `src/elspeth/engine/executors.py` - Added `_batch_ids` dict (line 467), `get_batch_id()` helper (line 474), deleted AggregationLike
- `src/elspeth/engine/orchestrator.py` - RowPlugin alias at line 29
- `tests/engine/test_executors.py` - Updated assertions
- `tests/engine/test_processor.py` - Removed _batch_id from mocks

#### Verification
- [x] AggregationExecutor._batch_ids dict manages batch state
- [x] No AggregationLike in executors.py
- [x] orchestrator.py uses RowPlugin alias (line 29)
- [x] No aggregation._batch_id references remain
- [x] `mypy --strict` passes (Success: no issues)
- [x] All tests pass (167 engine tests)

---

### WP-05: Audit Schema Enhancement

**Status:** 🟢 Complete (2026-01-18)
**Plan:** [2026-01-17-wp05-audit-schema-enhancement.md](./2026-01-17-wp05-audit-schema-enhancement.md)
**Goal:** Add missing columns and fix types for audit completeness
**Unlocks:** WP-06

#### Files Modified
- `src/elspeth/contracts/enums.py` - Added TriggerType enum
- `src/elspeth/contracts/__init__.py` - Export TriggerType
- `src/elspeth/core/landscape/schema.py` - Added idempotency_key, trigger_type columns
- `src/elspeth/core/landscape/models.py` - Added fields, fixed Batch.status type
- `tests/contracts/test_enums.py` - Added TriggerType tests
- `tests/core/landscape/test_schema.py` - Added schema/model tests

#### Tasks
- [x] Task 1: Add TriggerType enum (5 values: COUNT, TIMEOUT, CONDITION, END_OF_SOURCE, MANUAL)
- [x] Task 2: Add idempotency_key to artifacts table (String(256))
- [x] Task 3: Add trigger_type to batches table (String(32))
- [x] Task 4: Fix Batch.status type from str to BatchStatus
- [x] Task 5: Generate Alembic migration (SKIPPED - pre-release, no DB to migrate)
- [x] Task 6: Run full verification

#### Commits
```
e4f0a9c feat(contracts): add TriggerType enum for aggregation triggers
2af8349 feat(landscape): add idempotency_key to artifacts table
df2fa70 feat(landscape): add trigger_type to batches table
7d8df17 fix(models): change Batch.status from str to BatchStatus
```

#### Verification ✅
- [x] `TriggerType` enum exists with 5 values (COUNT, TIMEOUT, CONDITION, END_OF_SOURCE, MANUAL)
- [x] `TriggerType` exported from `elspeth.contracts`
- [x] `artifacts_table` has `idempotency_key` column (String(256))
- [x] `Artifact` model has `idempotency_key` field
- [x] `batches_table` has `trigger_type` column (String(32))
- [x] `Batch` model has `trigger_type` field
- [x] `Batch.status` type is `BatchStatus` (not `str`)
- [x] `mypy --strict` passes on contracts and landscape modules
- [x] All 8 WP-05 tests pass (3 enum + 2 artifacts + 2 batches + 1 status type)

---

### WP-06: Aggregation Triggers

**Status:** 🟢 Complete (2026-01-18)
**Plan:** [2026-01-18-wp06-aggregation-triggers.md](./2026-01-18-wp06-aggregation-triggers.md)
**Goal:** Config-driven aggregation triggers replace plugin-driven decisions

#### Summary
Moved aggregation trigger logic from plugins to engine:
- TriggerConfig model (count, timeout_seconds, condition)
- AggregationSettings model (name, plugin, trigger, output_mode, options)
- TriggerEvaluator class for engine-controlled trigger evaluation
- Removed AcceptResult.trigger, BaseAggregation.should_trigger(), BaseAggregation.reset()
- Integrated into AggregationExecutor and RowProcessor

#### Tasks
- [x] Create `TriggerConfig` model in `src/elspeth/core/config.py`
- [x] Create `AggregationSettings` model in `src/elspeth/core/config.py`
- [x] Add `aggregations` field to `ElspethSettings`
- [x] Implement `TriggerEvaluator` in `src/elspeth/engine/triggers.py`:
  - [x] `count` trigger
  - [x] `timeout` trigger (via time.monotonic())
  - [x] `condition` trigger (via ExpressionParser)
  - [x] `end_of_source` trigger (implicit - engine calls flush at source exhaustion)
- [x] Remove `AcceptResult.trigger` field (BREAKING)
- [x] Remove `BaseAggregation.should_trigger()` and `reset()` (BREAKING)
- [x] Integrate `TriggerEvaluator` into `AggregationExecutor`
- [x] Update `RowProcessor` to use engine-controlled triggers

#### Verification
- [x] Config validation rejects invalid triggers (at least one required)
- [x] count, timeout, condition triggers work (15 TriggerEvaluator tests)
- [x] mypy --strict passes on all 6 modified source files
- [x] 532 affected tests pass

---

### WP-07: Fork Work Queue

**Status:** 🟢 Complete (2026-01-18)
**Goal:** Forked child tokens actually execute through their paths

#### Files Modified
- `src/elspeth/engine/processor.py` - Work queue using `collections.deque`, `_WorkItem` dataclass, BFS token processing
- `src/elspeth/engine/orchestrator.py` - Updated to iterate over `list[RowResult]` returns
- `tests/engine/test_processor.py` - Updated existing tests for new return type, added iteration guard and nested fork tests
- `tests/integration/test_fork_pipeline.py` - Full pipeline fork integration test

#### Tasks
- [x] Task 1: Implement `_WorkItem` dataclass tracking token and start_step
- [x] Task 2: Implement work queue using `collections.deque` for BFS processing
- [x] Task 3: Update `process_row()` to return `list[RowResult]` instead of single `RowResult`
- [x] Task 4: Add `MAX_WORK_QUEUE_ITERATIONS = 10,000` safety guard
- [x] Task 5: Add iteration guard test (raises `RuntimeError` on infinite loop)
- [x] Task 6: Add nested fork test (fork within fork)
- [x] Task 7: Add full pipeline integration test

#### Commits
```
6b95e00 feat(engine): implement work queue for fork child execution (WP-07 Tasks 1-2)
5ce491b test(processor): update existing tests for list[RowResult] return type (WP-07 Task 3)
da7cf5c test(integration): add full pipeline fork test (WP-07)
```

#### Verification ✅
- [x] Fork creates children that execute through remaining transforms
- [x] Each child follows its assigned path (BFS order)
- [x] Parent FORKED, children reach terminal states
- [x] Audit trail shows complete lineage via `parent_token_id`
- [x] MAX_WORK_QUEUE_ITERATIONS guard prevents infinite loops
- [x] Nested forks work correctly (fork within fork)
- [x] All tests pass

---

### WP-08: Coalesce Executor

**Status:** 🟢 Complete (2026-01-18)
**Plan:** [2026-01-18-wp08-coalesce-executor.md](./2026-01-18-wp08-coalesce-executor.md)
**Goal:** Merge tokens from parallel fork paths
**Blocked by:** WP-07 ✅

#### Files Created
- `src/elspeth/engine/coalesce_executor.py` (~350 lines) - CoalesceExecutor, CoalesceOutcome, _PendingCoalesce
- `tests/engine/test_coalesce_executor.py` (~500 lines) - 15 tests across 7 test classes

#### Files Modified
- `src/elspeth/core/config.py` - Added `CoalesceSettings` model + `coalesce` field in `ElspethSettings`
- `tests/core/test_config.py` - Added CoalesceSettings + ElspethSettings coalesce tests
- `src/elspeth/plugins/protocols.py` - Added `FIRST` to `CoalescePolicy` enum
- `src/elspeth/engine/__init__.py` - Exports for `CoalesceExecutor`, `CoalesceOutcome`

#### Tasks
- [x] Task 1: Add FIRST policy to CoalescePolicy enum
- [x] Task 2: Create CoalesceSettings config model with validators
- [x] Task 3: Create CoalesceExecutor skeleton (register_coalesce, get_registered_names)
- [x] Task 4: Implement accept() with REQUIRE_ALL policy
- [x] Task 5: Implement FIRST, QUORUM, BEST_EFFORT policies with check_timeouts()
- [x] Task 5.5: Record coalesce audit metadata (arrived branches, policy applied, timing)
- [x] Task 6: Export CoalesceExecutor from engine/__init__.py
- [x] Task 7: Add coalesce field to ElspethSettings
- [x] Task 8: Integration test - full fork/coalesce pipeline
- [x] Task 8.5: Add flush_pending() for graceful shutdown

#### Commits (13 total)
```
6447582 feat(protocols): add FIRST policy to CoalescePolicy enum (WP-08)
f864a3c feat(config): add CoalesceSettings for token merging configuration (WP-08)
ff25368 test(config): add negative value tests for CoalesceSettings
235c589 feat(engine): create CoalesceExecutor skeleton (WP-08 Task 3)
4798f50 refactor(test): clean up coalesce executor test fixtures
77ad866 feat(coalesce): implement accept() with require_all policy (WP-08 Task 4)
3591fc6 feat(coalesce): implement FIRST, QUORUM, BEST_EFFORT policies (WP-08 Task 5)
92745b2 test(coalesce): add missing coverage for check_timeouts edge cases
26f8eb1 feat(coalesce): record audit metadata for coalesce events (WP-08 Task 5.5)
1994df9 feat(engine): export CoalesceExecutor (WP-08 Task 6)
299610b feat(config): add coalesce field to ElspethSettings (WP-08 Task 7)
84499f3 test(coalesce): add fork/process/coalesce integration test (WP-08 Task 8)
68b8e0f feat(coalesce): add flush_pending for graceful shutdown (WP-08 Task 8.5)
```

#### Verification ✅
- [x] All 4 policies work (require_all, quorum, best_effort, first)
- [x] All 3 merge strategies work (union, nested, select)
- [x] Timeout handling works (uses `time.monotonic()`)
- [x] Audit metadata recorded (arrived branches, policy, timing)
- [x] CoalesceSettings validated (policy/quorum/select_branch requirements)
- [x] Integration test: fork → parallel transforms → coalesce
- [x] flush_pending() for graceful shutdown
- [x] All tests pass
- [x] mypy --strict passes

#### Architecture Notes
- **NodeStateStatus vs RowOutcome:** NodeStateStatus (OPEN, COMPLETED, FAILED) stored in DB; RowOutcome (COALESCED, etc.) derived at query time from node_states + routing_events + ancestry
- **Token correlation:** Tokens correlated by row_id (same source row that was forked)
- **Timeout handling:** Uses `time.monotonic()` for accurate elapsed time measurement

---

### WP-09: Engine-Level Gates

**Status:** 🟢 Complete (2026-01-18)
**Goal:** Gates become config-driven engine operations with safe expression parsing
**Dependency:** WP-02 ✅

#### Files Created
- `src/elspeth/engine/expression_parser.py` (424 lines) - Safe AST-based expression parser
- `tests/engine/test_expression_parser.py` (961 lines) - Unit tests + 2000+ fuzz inputs
- `tests/engine/test_engine_gates.py` (1088 lines) - 22 integration tests

#### Files Modified
- `src/elspeth/core/config.py` - Added `GateSettings` model with condition validation
- `src/elspeth/engine/executors.py` - Added `execute_config_gate()` method
- `src/elspeth/engine/orchestrator.py` - Pipeline integration for config gates
- `src/elspeth/engine/processor.py` - Config gate processing in row loop
- `src/elspeth/core/dag.py` - Config gate ID map and route resolution
- `src/elspeth/engine/__init__.py` - Exports for new classes

#### Tasks
- [x] Task 1: Create `src/elspeth/engine/expression_parser.py`
  - [x] Implement safe AST-based expression evaluation (NOT Python eval)
  - [x] Allow: field access, comparisons, boolean operators, membership, literals
  - [x] Reject: function calls, imports, attribute access, assignment, lambda, comprehensions
  - [x] Fix: Reject starred expressions (*) and dict spread (**)
- [x] Task 2: Expression Parser Security Tests with Fuzz Testing
  - [x] 2,277+ fuzz inputs (Hypothesis + deterministic seeded)
  - [x] All attack patterns rejected at parse time
- [x] Task 3: Create `GateSettings` in config.py
  - [x] Pydantic model with condition, routes, fork_to fields
  - [x] Condition validated by ExpressionParser at config load time
  - [x] Route destinations validated (continue, fork, or sink name)
- [x] Task 4: Add `execute_config_gate()` to GateExecutor
  - [x] Evaluates conditions using ExpressionParser
  - [x] Boolean results → "true"/"false" labels
  - [x] Full audit trail recording
- [x] Task 5: Update Orchestrator for engine-level gates
  - [x] PipelineConfig.gates field
  - [x] Config gate node registration in Landscape
  - [x] Route resolution map pre-computation
- [x] Task 6: Integration tests for engine gates
  - [x] 22 tests covering all WP-09 verification requirements

#### Commits (8 total)
```
3e1a127 feat(engine): add safe AST-based expression parser for gate conditions
39e13b9 fix(expression_parser): reject starred and dict spread expressions at parse time
666e88c test(expression_parser): add fuzz testing with 2000+ random inputs (WP-09)
23f2537 feat(config): add GateSettings for engine-level config-driven routing (WP-09)
3577ad6 refactor(config): consolidate GateSettings route validators
ae56d02 feat(engine): add execute_config_gate for config-driven gate evaluation (WP-09 Task 4)
a7d2099 feat(engine): integrate config-driven gates into orchestrator pipeline (WP-09 Task 5)
c424826 test(engine): add comprehensive integration tests for engine-level gates
```

#### Verification ✅
- [x] Expression parser rejects all unsafe code (8 attack patterns tested)
- [x] Composite conditions work: `row['a'] > 0 and row['b'] == 'x'`
- [x] fork_to creates child tokens (config-level verified, execution in WP-07)
- [x] Route labels resolve correctly
- [x] Fuzz testing: 2,277+ inputs, no crashes, no code execution
- [x] All 383 tests pass (engine + config)
- [x] mypy --strict passes

---

### WP-10: Quarantine Implementation

**Status:** 🟢 Complete (2026-01-18)
**Plan:** [2026-01-18-wp10-quarantine-implementation.md](./2026-01-18-wp10-quarantine-implementation.md)
**Goal:** QUARANTINED terminal state becomes reachable for transform errors
**Blocked by:** WP-07 ✅

#### Architecture
When a transform returns `TransformResult.error()` with `_on_error` configured:
- `_on_error = "discard"` → `RowOutcome.QUARANTINED` (intentional rejection)
- `_on_error = sink_name` → `RowOutcome.ROUTED` (error sent to designated sink)

This is achieved by adding `error_sink` as third return value from `execute_transform()`.

#### Files Modified
- `src/elspeth/engine/executors.py` - Return 3-tuple from `execute_transform()`
- `src/elspeth/engine/processor.py` - Handle error_sink for QUARANTINED/ROUTED outcomes
- `src/elspeth/engine/orchestrator.py` - Add `rows_quarantined` metric to `RunResult`
- `tests/engine/test_executors.py` - 3 new tests for error_sink return value
- `tests/engine/test_processor.py` - 2 new outcome tests + 2 integration tests
- `tests/engine/test_orchestrator.py` - 1 new test for quarantine metrics

#### Tasks
- [x] Task 1: Update `TransformExecutor.execute_transform()` to return 3-tuple
- [x] Task 2: Update Processor to return QUARANTINED/ROUTED based on error_sink
- [x] Task 3: Update existing tests for new return signature (done in Task 1)
- [x] Task 4: Add `rows_quarantined` metric to `RunResult` in Orchestrator
- [x] Task 5: Integration tests for full quarantine flow
- [x] Task 6: Update tracker

#### Commits
```
df6917c feat(executor): return error_sink from execute_transform (WP-10 Task 1)
602f785 feat(processor): return QUARANTINED/ROUTED for transform errors (WP-10 Task 2)
8bd7b2b feat(orchestrator): add rows_quarantined metric (WP-10 Task 4)
e9a6029 test(processor): add quarantine integration tests (WP-10 Task 5)
```

#### Verification ✅
- [x] `TransformExecutor.execute_transform()` returns 3-tuple: `(result, token, error_sink)`
- [x] `error_sink` is None (success), "discard" (quarantine), or sink_name (routed)
- [x] Processor returns `QUARANTINED` when `error_sink == "discard"`
- [x] Processor returns `ROUTED` with sink_name when `error_sink` is a sink name
- [x] Orchestrator counts `rows_quarantined` separately from `rows_failed`
- [x] Pipeline continues processing after quarantine (doesn't crash)
- [x] Audit trail records quarantined rows (node_state with status="failed")
- [x] All existing tests pass after signature update
- [x] `mypy --strict` passes
- [x] All tests pass

---

### WP-11: Orphaned Code Cleanup

**Status:** 🟢 Complete (2026-01-18)
**Plan:** [2026-01-17-wp11-orphaned-code-cleanup.md](./2026-01-17-wp11-orphaned-code-cleanup.md)
**Goal:** Remove dead code and defensive programming patterns, KEEP audit-critical infrastructure

#### Decisions Made
- **RetryManager:** KEEP & INTEGRATE (Phase 5)
- **Call infrastructure:** KEEP (Phase 6)
- **on_register():** DELETE (never called)
- **Defensive getattr in manager.py:** DELETE (hides protocol violations) - added 2026-01-18
- **Non-Pydantic schema check:** CRASH instead of silent None - added 2026-01-18
- **TUI .get() defaults:** DELETE (masks incomplete data, violates tui/types.py) - added 2026-01-18
- **TUI silent exceptions:** LOG before returning failed states - added 2026-01-18

#### Tasks
- [x] Task 1: Remove on_register() from 4 base classes
- [x] Task 2: Verify RetryManager is ready for integration
- [x] Task 3: Verify Call infrastructure is intact
- [x] Task 4: Remove defensive `getattr(cls, "name", cls.__name__)` from manager.py (6 occurrences)
- [x] Task 5: Fix `_schema_hash()` to crash on non-Pydantic schemas
- [x] Task 6: Fix TUI node_detail.py to not use .get() defaults
- [x] Task 7: Fix TUI exception handlers to log instead of silently swallow
- [x] Task 8: Run full verification

#### Verification
- [x] `on_register()` removed from 4 base classes
- [x] Defensive getattr for plugin names removed (6 occurrences)
- [x] `_schema_hash()` crashes on non-Pydantic schemas
- [x] TUI uses direct field access (not .get() defaults)
- [x] TUI logs exceptions before returning failed states
- [x] RetryManager tests pass
- [x] Call infrastructure intact
- [x] All tests pass (plugins, engine, TUI)

---

### WP-11.99: Config-Driven Plugin Schemas

**Status:** 🟢 Complete
**Plan:** [2026-01-17-wp11.99-config-driven-schemas.md](./2026-01-17-wp11.99-config-driven-schemas.md)
**Goal:** Replace hardcoded schemas with mandatory config-driven definitions
**Unlocks:** WP-12 ✅

**Architecture:** Every data-processing plugin must declare `schema` in config:
- `fields: dynamic` - Accept anything (logged for audit)
- Explicit fields with `mode: strict` (exactly these) or `mode: free` (at least these)

**Trust Boundaries:**
| Plugin | On Schema Violation |
|--------|---------------------|
| Source | Quarantine row, continue (THEIR DATA) |
| Transform | Crash (OUR CODE bug) |
| Sink | Crash (transform bug) |

#### Tasks
- [ ] Task 1: Create SchemaConfig and FieldDefinition in contracts/schema.py
- [ ] Task 2: Create schema factory in plugins/schema_factory.py
- [ ] Task 3: Add DataPluginConfig with required schema
- [ ] Task 4: Add schema recording to landscape recorder
- [ ] Task 5: Update source plugins (csv, json)
- [ ] Task 6: Update sink plugins (csv, json, database)
- [ ] Task 7: Update transform plugins (field_mapper, passthrough)
- [ ] Task 8: Run full verification

#### Verification
- [ ] SchemaConfig and FieldDefinition types created
- [ ] Schema factory creates Pydantic models from config
- [ ] All data plugins require schema in config
- [ ] Schema choices recorded in audit trail (nodes table)
- [ ] Source validates + coerces at boundary
- [ ] No hardcoded `extra="allow"` schemas in plugin files
- [ ] All tests pass
- [ ] mypy --strict passes

---

### WP-12: Utility Consolidation

**Status:** 🟢 Complete (2026-01-18)
**Plan:** [2026-01-17-wp12-utility-consolidation.md](./2026-01-17-wp12-utility-consolidation.md)
**Goal:** Extract `get_nested_field()` utility to shared module
**Blocked by:** WP-11.99 ✅

> **Note:** Schema consolidation is handled by WP-11.99. This WP only extracts the `_get_nested()` utility.

#### Tasks
- [x] Task 1: Create utils.py with get_nested_field()
- [x] Task 2: Update field_mapper.py to use shared utility
- [x] Task 3: Run verification

#### Verification
- [x] `get_nested_field()` has tests (9 tests pass)
- [x] `field_mapper.py` imports from utils, no local `_get_nested`
- [x] field_mapper tests pass (14 tests pass)

---

### WP-13: Sink Test Rewrites

**Status:** 🟢 Complete (done implicitly with WP-03)
**Goal:** All sink tests use batch signature
**Blocked by:** WP-03 ✅, WP-04 ✅

**Note:** Tests were updated alongside WP-03 sink implementation changes. This is best practice - changing implementation and tests together.

#### Tasks
- [x] Rewrite `tests/plugins/sinks/test_csv_sink.py` (done with WP-03)
- [x] Rewrite `tests/plugins/sinks/test_json_sink.py` (done with WP-03)
- [x] Rewrite `tests/plugins/sinks/test_database_sink.py` (done with WP-03)
- [x] Create MockSink fixture for engine tests that need it

#### Verification
- [x] All sink plugin tests pass (41 tests)
- [x] Batch signature: `sink.write([...], ctx)` throughout
- [x] ArtifactDescriptor assertions present
- [x] content_hash and size_bytes verified

---

### WP-14: Engine Test Rewrites

**Status:** 🔴 Not Started
**Goal:** Engine tests updated for all architectural changes
**Blocked by:** WP-06, WP-07, WP-08, WP-09, WP-10

#### Files to Update
- [ ] `tests/engine/test_processor.py` (828 lines) - Fork work queue, coalesce, quarantine
- [ ] `tests/engine/test_executors.py` (1956 lines) - Aggregation triggers, gate routing
- [ ] `tests/engine/test_orchestrator.py` (3920+ lines) - Engine gates, route resolution
- [ ] `tests/engine/test_integration.py` (1048 lines) - End-to-end with new architecture
- [ ] `tests/plugins/test_integration.py` (237 lines) - Plugin integration

#### Verification
- [ ] All tests pass
- [ ] Coverage maintained
- [ ] No references to old patterns

---

### WP-15: RetryManager Integration

**Status:** 🟢 Complete (2026-01-18)
**Plan:** [2026-01-18-wp15-retry-manager-integration.md](./2026-01-18-wp15-retry-manager-integration.md)
**Goal:** Integrate existing RetryManager into transform execution with full audit trail
**Dependencies:** None (independent)

**Context:** RetryManager exists at `src/elspeth/engine/retry.py` but was never wired into the engine. This WP integrates it so transient failures (network timeouts, rate limits) are automatically retried with each attempt recorded in the audit trail.

#### Files Modified
- `src/elspeth/engine/retry.py` - Added `RetryConfig.from_settings()` factory method
- `src/elspeth/engine/executors.py` - Added `attempt: int = 0` parameter to `execute_transform()`
- `src/elspeth/engine/processor.py` - Added `retry_manager` parameter, `_execute_transform_with_retry()` method
- `src/elspeth/contracts/results.py` - Added `error: dict[str, Any] | None` field to `RowResult`
- `src/elspeth/engine/orchestrator.py` - Wire RetryManager creation from settings

#### Files Created
- `tests/integration/test_retry_integration.py` - 3 integration tests proving retry audit trail

#### Tasks
- [x] Task 1: Add RetryConfig.from_settings() factory
- [x] Task 2: Add attempt parameter to execute_transform
- [x] Task 3: Add RetryManager to RowProcessor
- [x] Task 4: Implement retry wrapper for transform execution
- [x] Task 5: Handle MaxRetriesExceeded with FAILED outcome
- [x] Task 6: Wire RetryManager in Orchestrator
- [x] Task 7: Integration test for retry audit trail
- [x] Task 8: Update tracker

#### Commits
```
443114a feat(retry): add RetryConfig.from_settings() factory (WP-15 Task 1)
0242ef1 feat(executor): add attempt parameter to execute_transform (WP-15 Task 2)
9f09af2 feat(processor): add retry_manager parameter to RowProcessor (WP-15 Task 3)
60e2177 feat(processor): implement retry wrapper for transform execution (WP-15 Task 4)
41c2fdc feat(processor): handle MaxRetriesExceeded with FAILED outcome (WP-15 Task 5)
5814de7 feat(orchestrator): wire RetryManager from settings (WP-15 Task 6)
f415e1c test(integration): add retry audit trail integration tests (WP-15 Task 7)
8f3ed16 style(tests): prefix unused variables with underscore
```

#### Verification ✅
- [x] RetryConfig.from_settings() creates config from Pydantic model
- [x] execute_transform accepts and passes attempt number
- [x] Transient exceptions (ConnectionError, TimeoutError, OSError) are retried
- [x] MaxRetriesExceeded returns FAILED outcome with error details
- [x] Each attempt recorded as separate node_state with attempt number
- [x] RowResult.error captures exception type, message, attempts, last_error
- [x] Integration tests prove: attempts 0,1,2 each recorded in node_states
- [x] All tests pass (3 integration tests + existing unit tests)
- [x] mypy --strict passes
- [x] ruff lint clean

---

## Risk Register

| WP | Risk | Likelihood | Impact | Mitigation |
|----|------|------------|--------|------------|
| WP-03 | Content hashing edge cases | Medium | Medium | Test with large files, binary data |
| WP-07 | Infinite loops in work queue | Low | High | Max iteration guard |
| WP-08 | Timeout race conditions | Medium | Medium | Use monotonic clock |
| WP-09 | Expression parser security | ✅ Mitigated | High | AST whitelist validation, 2277+ fuzz inputs, 8 attack patterns rejected |
| WP-14 | Large test rewrite scope | High | Medium | Incremental, focus on critical paths |
| WP-15 | Retry storms under load | Low | Medium | max_delay cap (60s), bounded max_attempts |

---

## Change Log

| Date | WP | Change | Author |
|------|-----|--------|--------|
| 2026-01-17 | — | Created tracking document | Claude |
| 2026-01-17 | WP-01 | ✅ Completed - protocols and base classes aligned | — |
| 2026-01-17 | WP-03 | ✅ Completed - sinks return ArtifactDescriptor | — |
| 2026-01-17 | WP-04 | Created detailed plan: wp04-sink-adapter-update.md | Claude |
| 2026-01-17 | WP-06 | Added stale code cleanup (AcceptResult.trigger, should_trigger, reset) | Claude |
| 2026-01-17 | WP-11 | Decision: KEEP RetryManager, KEEP Call infrastructure for audit | Claude |
| 2026-01-17 | WP-14 | Added note to split into WP-14a/b/c/d/e when executed | Claude |
| 2026-01-17 | WP-04a | **NEW**: Added WP-04a to delete TransformLike/GateLike/AggregationLike (from paused interface-unification plan) | Claude |
| 2026-01-17 | — | Resequenced sprints: WP-02 + WP-09 now in Sprint 4 (no gate gap) | Claude |
| 2026-01-17 | WP-04 | Fixed: use is_batch_sink() instead of runtime_checkable Protocol | Claude |
| 2026-01-17 | WP-12 | Created detailed plan: wp12-utility-consolidation.md | Claude |
| 2026-01-17 | WP-12 | Fixed: Task 4 (DynamicSchema in sinks) now required, not optional | Claude |
| 2026-01-17 | WP-03 | ✅ Verified: 41 tests pass, mypy clean, all checklist items confirmed | Claude |
| 2026-01-17 | WP-04 | 🟢 READY: Dependencies satisfied (WP-03), plan reviewed against codebase | Claude |
| 2026-01-17 | WP-12 | 🟢 READY: No blockers, sentinels.py exists, field_mapper.py has _get_nested | Claude |
| 2026-01-17 | WP-05 | Created detailed plan: wp05-audit-schema-enhancement.md | Claude |
| 2026-01-17 | WP-11 | Created detailed plan: wp11-orphaned-code-cleanup.md | Claude |
| 2026-01-17 | WP-04 | **MAJOR FIX**: Changed from "update adapter" to "delete adapter & SinkLike" | Claude |
| 2026-01-17 | WP-13 | Fixed: Removed test_adapters.py reference (deleted in WP-04) | Claude |
| 2026-01-17 | WP-14 | Fixed: Removed WP-14a (sink adapter tests) - no longer exists | Claude |
| 2026-01-17 | — | Fixed work-packages.md: WP-04, WP-13, WP-14 all updated | Claude |
| 2026-01-17 | WP-04a | Created detailed plan with Option C: batch state internal to executor | Claude |
| 2026-01-17 | WP-04a | TransformLike/GateLike already deleted (f08c19a), only AggregationLike remains | Claude |
| 2026-01-18 | WP-02 | ✅ **COMPLETE** - 9 files deleted, 9 modified, 11 commits. Plan gap fixed (test_run_with_row_plugins.py). Ready for WP-09. | Claude |
| 2026-01-18 | WP-09 | ✅ **COMPLETE** - 8 commits, 6 tasks. Expression parser (424 lines), fuzz testing (2277+ inputs), GateSettings config, execute_config_gate(), orchestrator integration, 22 integration tests. All verification requirements met. | Claude |
| 2026-01-18 | WP-07 | ✅ **COMPLETE** - Work queue using `collections.deque` for BFS token processing, `_WorkItem` dataclass, `process_row()` returns `list[RowResult]`, MAX_WORK_QUEUE_ITERATIONS=10,000 safety guard, nested fork support. Unlocks WP-08 and WP-10. | Claude |
| 2026-01-18 | WP-08 | ✅ **COMPLETE** - 13 commits, 10 tasks. CoalesceExecutor (350 lines), CoalesceSettings config, all 4 policies (require_all, quorum, best_effort, first), all 3 merge strategies (union, nested, select), flush_pending(), audit metadata recording. Unlocks WP-14. | Claude |
| 2026-01-18 | WP-10 | ✅ **COMPLETE** - 4 commits, 6 tasks. `execute_transform()` returns 3-tuple with `error_sink`, processor returns QUARANTINED/ROUTED based on error routing, `rows_quarantined` metric added to RunResult, integration tests for full quarantine flow. Unlocks WP-14. | Claude |
| 2026-01-18 | WP-05 | ✅ **COMPLETE** - 4 commits, 6 tasks. TriggerType enum (5 values), idempotency_key column (String(256)), trigger_type column (String(32)), Batch.status typed as BatchStatus. Alembic migration skipped (pre-release). Unlocks WP-06. | Claude |
| 2026-01-18 | WP-11 | Added Task 4: Remove defensive `getattr(cls, "name", cls.__name__)` from manager.py (6 occurrences). Found during defensive programming audit - all protocols require `name: str`, fallback hides violations. | Claude |
| 2026-01-18 | WP-11 | Added Tasks 5-7: (5) Crash on non-Pydantic schemas in `_schema_hash()`, (6) Remove TUI `.get()` defaults that mask incomplete data, (7) Log TUI exceptions instead of silently swallowing. All found during defensive programming deep-dive. | Claude |
| 2026-01-18 | WP-11 | ✅ **COMPLETE** - 7 commits, 8 tasks. Removed `on_register()` from 4 base classes, removed defensive getattr (6 occurrences), fixed `_schema_hash()` to crash on non-Pydantic, fixed TUI `.get()` patterns, added exception logging. Updated PHASE3_INTEGRATION.md. 297 plugin tests, 348 engine tests, 43 TUI tests all pass. | Claude |
| 2026-01-18 | WP-15 | **CREATED** - New work package to integrate RetryManager into transform execution. Plan at `2026-01-18-wp15-retry-manager-integration.md`. 8 tasks covering: factory method, attempt tracking, RowProcessor integration, MaxRetriesExceeded handling, Orchestrator wiring. Independent - can run anytime. | Claude |
| 2026-01-18 | WP-15 | ✅ **COMPLETE** - 8 commits, 8 tasks. RetryConfig.from_settings() factory, attempt parameter to execute_transform(), retry wrapper in RowProcessor, MaxRetriesExceeded → FAILED outcome with error details, Orchestrator wiring, 3 integration tests proving attempts 0,1,2 recorded in node_states. Added `error: dict[str, Any] \| None` to RowResult for failure details. | Claude |
| | | | |
