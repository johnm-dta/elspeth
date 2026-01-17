# Plugin Refactor Progress Tracker

> **Created:** 2026-01-17
> **Source:** work-packages.md, gap-analysis.md
> **Contract:** plugin-protocol.md v1.1
> **Total Effort:** ~69 hours

---

## Quick Status

| WP | Name | Status | Effort | Dependencies | Unlocks |
|----|------|--------|--------|--------------|---------|
| WP-01 | Protocol & Base Class Alignment | 🟢 Complete | 2h | None | WP-03 |
| WP-02 | Gate Plugin Deletion | 🔴 Not Started | 1h | None | — |
| WP-03 | Sink Implementation Rewrite | 🔴 Not Started | 4h | WP-01 | WP-04, WP-13 |
| WP-04 | Sink Adapter Update | 🔴 Not Started | 1h | WP-03 | WP-13 |
| WP-05 | Audit Schema Enhancement | 🔴 Not Started | 2h | None | WP-06 |
| WP-06 | Aggregation Triggers | 🔴 Not Started | 6h | WP-05 | WP-14 |
| WP-07 | Fork Work Queue | 🔴 Not Started | 8h | None | WP-08, WP-10 |
| WP-08 | Coalesce Executor | 🔴 Not Started | 8h | WP-07 | WP-14 |
| WP-09 | Engine-Level Gates | 🔴 Not Started | 10h | (after WP-02) | WP-14 |
| WP-10 | Quarantine Implementation | 🔴 Not Started | 4h | WP-07 | WP-14 |
| WP-11 | Orphaned Code Cleanup | 🔴 Not Started | 2h | None | — |
| WP-12 | Utility Consolidation | 🔴 Not Started | 1h | (after WP-02) | — |
| WP-13 | Sink Test Rewrites | 🔴 Not Started | 4h | WP-03, WP-04 | — |
| WP-14 | Engine Test Rewrites | 🔴 Not Started | 16h | WP-06,07,08,09,10 | — |

**Legend:** 🔴 Not Started | 🟡 In Progress | 🟢 Complete | ⏸️ Blocked

---

## Dependency Graph

```
WP-01 ──┬──► WP-03 ──► WP-04 ──► WP-13
        │
WP-02   │   (independent)
        │
WP-05 ──┴──► WP-06
                        ╲
WP-07 ──┬──► WP-08 ──────┬──► WP-14
        └──► WP-10 ──────╱

WP-09 ──────────────────╱

WP-11       (independent)
WP-12       (independent, after WP-02)
```

---

## Sprint Allocation

### Sprint 1: Foundation & Cleanup
- [x] WP-01: Protocol & Base Class Alignment
- [ ] WP-02: Gate Plugin Deletion
- [ ] WP-05: Audit Schema Enhancement
- [ ] WP-11: Orphaned Code Cleanup

### Sprint 2: Sink Contract
- [ ] WP-03: Sink Implementation Rewrite
- [ ] WP-04: Sink Adapter Update
- [ ] WP-12: Utility Consolidation
- [ ] WP-13: Sink Test Rewrites

### Sprint 3: DAG & Aggregation
- [ ] WP-06: Aggregation Triggers
- [ ] WP-07: Fork Work Queue
- [ ] WP-10: Quarantine Implementation

### Sprint 4: Advanced Features
- [ ] WP-08: Coalesce Executor
- [ ] WP-09: Engine-Level Gates

### Sprint 5: Verification
- [ ] WP-14: Engine Test Rewrites
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

**Status:** 🔴 Not Started
**Goal:** Complete removal of plugin-based gates (engine gates come in WP-09)

#### Files to DELETE
- [ ] `src/elspeth/plugins/gates/filter_gate.py` (249 lines)
- [ ] `src/elspeth/plugins/gates/field_match_gate.py` (193 lines)
- [ ] `src/elspeth/plugins/gates/threshold_gate.py` (144 lines)
- [ ] `src/elspeth/plugins/gates/hookimpl.py` (22 lines)
- [ ] `src/elspeth/plugins/gates/__init__.py` (11 lines)
- [ ] `tests/plugins/gates/test_filter_gate.py` (276 lines)
- [ ] `tests/plugins/gates/test_field_match_gate.py` (230 lines)
- [ ] `tests/plugins/gates/test_threshold_gate.py` (221 lines)
- [ ] `tests/plugins/gates/__init__.py` (1 line)

#### Files to MODIFY
- [ ] `src/elspeth/cli.py` - Remove gate imports (line 228) and registry (241-245)
- [ ] `src/elspeth/plugins/manager.py` - Remove builtin_gates import (161) and registration (168)
- [ ] `tests/plugins/test_base.py` - Remove ThresholdGate tests (74-100)
- [ ] `tests/plugins/test_protocols.py` - Remove ThresholdGate conformance (145-191)

#### Verification
- [ ] `grep -r "FilterGate\|FieldMatchGate\|ThresholdGate" src/` returns nothing
- [ ] No imports of deleted gate plugins anywhere
- [ ] Tests pass

---

### WP-03: Sink Implementation Rewrite

**Status:** 🔴 Not Started
**Goal:** All sinks conform to batch signature with ArtifactDescriptor return
**Blocked by:** WP-01

#### Sinks to Rewrite
- [ ] `src/elspeth/plugins/sinks/csv_sink.py`
  - [ ] Change `write(row) -> None` to `write(rows) -> ArtifactDescriptor`
  - [ ] Implement SHA-256 content hashing of written file
  - [ ] Add `determinism = Determinism.IO_WRITE`
  - [ ] Add `plugin_version = "1.0.0"`
  - [ ] Add `on_start()` and `on_complete()` lifecycle hooks

- [ ] `src/elspeth/plugins/sinks/json_sink.py`
  - [ ] Change `write(row) -> None` to `write(rows) -> ArtifactDescriptor`
  - [ ] Implement SHA-256 content hashing of written file
  - [ ] Add `determinism = Determinism.IO_WRITE`
  - [ ] Add `plugin_version = "1.0.0"`
  - [ ] Add `on_start()` and `on_complete()` lifecycle hooks

- [ ] `src/elspeth/plugins/sinks/database_sink.py`
  - [ ] Change `write(row) -> None` to `write(rows) -> ArtifactDescriptor`
  - [ ] Implement SHA-256 of canonical JSON payload before INSERT
  - [ ] Add `determinism = Determinism.IO_WRITE`
  - [ ] Add `plugin_version = "1.0.0"`
  - [ ] Add `on_start()` and `on_complete()` lifecycle hooks

#### Verification
- [ ] All sinks return ArtifactDescriptor
- [ ] content_hash is non-empty SHA-256
- [ ] size_bytes > 0 for non-empty writes
- [ ] Mypy passes

---

### WP-04: Sink Adapter Update

**Status:** 🔴 Not Started
**Goal:** SinkAdapter delegates to batch write, removes per-row loop
**Blocked by:** WP-03

#### Tasks
- [ ] Update `src/elspeth/engine/adapters.py`
  - [ ] Remove per-row loop in write (lines 183-185)
  - [ ] Direct delegation to `sink.write(rows)`
  - [ ] Remove `_rows_written` counter (lines 139-140)
  - [ ] Capture and return ArtifactDescriptor

#### Verification
- [ ] `SinkAdapter.write()` returns ArtifactDescriptor
- [ ] No per-row iteration in adapter
- [ ] Integration with orchestrator works

---

### WP-05: Audit Schema Enhancement

**Status:** 🔴 Not Started
**Goal:** Add missing columns and fix types for audit completeness

#### Tasks
- [ ] Add `idempotency_key` column to `artifacts` table (schema.py)
- [ ] Add `trigger_type` column to `batches` table (schema.py)
- [ ] Create `TriggerType` enum in `contracts/enums.py`
- [ ] Fix `Batch.status` type: `str` → `BatchStatus` (models.py:268)
- [ ] Update model dataclasses to match schema
- [ ] Generate Alembic migration

#### Verification
- [ ] Alembic migration generated
- [ ] Models match schema
- [ ] Mypy passes on contracts

---

### WP-06: Aggregation Triggers

**Status:** 🔴 Not Started
**Goal:** Config-driven aggregation triggers replace plugin-driven decisions
**Blocked by:** WP-05

#### Tasks
- [ ] Create `AggregationSettings` in `src/elspeth/core/config.py`
- [ ] Implement trigger types in orchestrator:
  - [ ] `count` trigger
  - [ ] `timeout` trigger
  - [ ] `condition` trigger
  - [ ] `end_of_source` trigger (implicit)
- [ ] Implement output modes:
  - [ ] `single` mode
  - [ ] `passthrough` mode
  - [ ] `transform` mode
- [ ] Move trigger evaluation from plugin to engine

#### Verification
- [ ] Config validation rejects invalid triggers
- [ ] All 4 trigger types work
- [ ] All 3 output modes work

---

### WP-07: Fork Work Queue

**Status:** 🔴 Not Started
**Goal:** Forked child tokens actually execute through their paths

#### Tasks
- [ ] Implement work queue in `src/elspeth/engine/processor.py`
- [ ] Replace single-pass execution with queue loop
- [ ] Process fork children through assigned paths
- [ ] Add max iteration guard (prevent infinite loops)

#### Verification
- [ ] Fork creates children that execute
- [ ] Each child follows its assigned path
- [ ] Parent FORKED, children reach terminal states
- [ ] Audit trail shows complete lineage

---

### WP-08: Coalesce Executor

**Status:** 🔴 Not Started
**Goal:** Merge tokens from parallel fork paths
**Blocked by:** WP-07

#### Tasks
- [ ] Create `src/elspeth/engine/coalesce_executor.py`
- [ ] Implement policies:
  - [ ] `require_all` - Wait for all branches
  - [ ] `quorum` - Wait for N branches
  - [ ] `best_effort` - Merge whatever arrives
  - [ ] `first` - Take first arrival
- [ ] Implement merge strategies:
  - [ ] `union` - Combine all fields
  - [ ] `nested` - Each branch as nested object
  - [ ] `select` - Take specific branch output
- [ ] Add coalesce handling to processor.py
- [ ] Export from `engine/__init__.py`

#### Verification
- [ ] COALESCED terminal state reachable
- [ ] All 4 policies work
- [ ] All 3 merge strategies work
- [ ] Timeout handling works

---

### WP-09: Engine-Level Gates

**Status:** 🔴 Not Started
**Goal:** Gates become config-driven engine operations with safe expression parsing
**Recommended after:** WP-02

#### Tasks
- [ ] Create `src/elspeth/engine/expression_parser.py`
  - [ ] Implement safe expression evaluation (NOT Python eval)
  - [ ] Allow: field access, comparisons, boolean operators, membership, literals
  - [ ] Reject: function calls, imports, attribute access, assignment, lambda
- [ ] Create `GateSettings` in config.py
- [ ] Refactor route resolution from GateExecutor to Orchestrator
- [ ] Simplify GateExecutor to only evaluate conditions

#### Verification
- [ ] Expression parser rejects unsafe code
- [ ] Composite conditions work: `row['a'] > 0 and row['b'] == 'x'`
- [ ] fork_to creates child tokens
- [ ] Route labels resolve correctly

---

### WP-10: Quarantine Implementation

**Status:** 🔴 Not Started
**Goal:** QUARANTINED terminal state becomes reachable
**Blocked by:** WP-07 (touches same file)

#### Tasks
- [ ] Add quarantine logic to `src/elspeth/engine/processor.py`
- [ ] Implement quarantine triggers:
  - [ ] Row fails schema validation
  - [ ] Required fields missing
  - [ ] Type coercion fails
  - [ ] External validation fails
- [ ] Record quarantine reason in audit trail

#### Verification
- [ ] QUARANTINED state reachable
- [ ] Quarantine reason recorded
- [ ] Pipeline continues after quarantine (doesn't crash)

---

### WP-11: Orphaned Code Cleanup

**Status:** 🔴 Not Started
**Goal:** Remove dead code that was never integrated

#### Tasks (DELETE)
- [ ] `engine/retry.py` lines 37-156 (RetryManager) - Decision: DELETE or INTEGRATE?
- [ ] `contracts/enums.py` lines 144-147 (CallType)
- [ ] `contracts/enums.py` lines 156-157 (CallStatus)
- [ ] `contracts/audit.py` lines 237-252 (Call dataclass)
- [ ] `landscape/recorder.py` lines 1707-1743 (get_calls())

#### Tasks (DEPRECATE)
- [ ] `plugins/base.py` lines 210-213 (should_trigger()) - Mark deprecated
- [ ] `plugins/base.py` lines 219-223 (reset()) - Mark deprecated
- [ ] `plugins/base.py` various (on_register()) - DELETE if never called

#### Verification
- [ ] No references to deleted code
- [ ] Tests pass
- [ ] No import errors

---

### WP-12: Utility Consolidation

**Status:** 🔴 Not Started
**Goal:** Extract duplicated code to shared utilities
**Recommended after:** WP-02

#### Tasks
- [ ] Create `src/elspeth/plugins/utils.py`
- [ ] Extract `_get_nested()` to `get_nested_field()`
- [ ] Update `field_mapper.py` to import from utils
- [ ] (Optional) Create `DynamicPluginSchema` factory

#### Verification
- [ ] `_get_nested` exists in only one location
- [ ] field_mapper.py works with imported utility

---

### WP-13: Sink Test Rewrites

**Status:** 🔴 Not Started
**Goal:** All sink tests use batch signature
**Blocked by:** WP-03, WP-04

#### Tasks
- [ ] Rewrite `tests/plugins/sinks/test_csv_sink.py`
- [ ] Rewrite `tests/plugins/sinks/test_json_sink.py`
- [ ] Rewrite `tests/plugins/sinks/test_database_sink.py`
- [ ] Update `tests/engine/test_adapters.py` (MockSink class)

#### Verification
- [ ] All sink tests pass
- [ ] Adapter tests pass
- [ ] No per-row write patterns remain

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

## Risk Register

| WP | Risk | Likelihood | Impact | Mitigation |
|----|------|------------|--------|------------|
| WP-03 | Content hashing edge cases | Medium | Medium | Test with large files, binary data |
| WP-07 | Infinite loops in work queue | Low | High | Max iteration guard |
| WP-08 | Timeout race conditions | Medium | Medium | Use monotonic clock |
| WP-09 | Expression parser security | Medium | High | Extensive fuzzing, AST-only parsing |
| WP-14 | Large test rewrite scope | High | Medium | Incremental, focus on critical paths |

---

## Change Log

| Date | WP | Change | Author |
|------|-----|--------|--------|
| 2026-01-17 | — | Created tracking document | Claude |
| 2026-01-17 | WP-01 | ✅ Completed - 4 commits, 32 tests passing, unlocks WP-03 | Claude |
| | | | |
