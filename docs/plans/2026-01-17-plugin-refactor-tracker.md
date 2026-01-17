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
| WP-02 | Gate Plugin Deletion | 🔴 Not Started | 1h | None | — |
| WP-03 | Sink Implementation Rewrite | 🟢 Complete | 4h | WP-01 | WP-04, WP-13 |
| WP-04 | Delete SinkAdapter & SinkLike | 🔴 Not Started | 2h | WP-03 | WP-04a, WP-13 |
| WP-04a | Delete *Like Protocol Duplications | 🔴 Not Started | 1h | WP-04 | — |
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
WP-01 ──┬──► WP-03 ──► WP-04 ──┬──► WP-04a
        │                      └──► WP-13
WP-02   │   (independent)

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

> **IMPORTANT:** WP-02 and WP-09 MUST be back-to-back (no gate gap)

### Sprint 1: Foundation
- [x] WP-01: Protocol & Base Class Alignment
- [ ] WP-05: Audit Schema Enhancement
- [ ] WP-11: Orphaned Code Cleanup (split into sub-tasks)

### Sprint 2: Sink Contract & Interface Cleanup
- [x] WP-03: Sink Implementation Rewrite
- [ ] WP-04: Delete SinkAdapter & SinkLike
- [ ] WP-04a: Delete *Like Protocol Duplications (TransformLike, GateLike, AggregationLike)
- [ ] WP-12: Utility Consolidation
- [ ] WP-13: Sink Test Rewrites

### Sprint 3: DAG & Aggregation
- [ ] WP-06: Aggregation Triggers (includes stale code cleanup)
- [ ] WP-07: Fork Work Queue
- [ ] WP-10: Quarantine Implementation

### Sprint 4: Gates & Coalesce
- [ ] WP-02: Gate Plugin Deletion ← Execute first
- [ ] WP-09: Engine-Level Gates ← Immediately after WP-02
- [ ] WP-08: Coalesce Executor

### Sprint 5: Verification
- [ ] WP-14: Engine Test Rewrites (split into WP-14a/b/c/d/e)
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

**Status:** 🔴 Not Started
**Plan:** [2026-01-17-wp04-sink-adapter-update.md](./2026-01-17-wp04-sink-adapter-update.md)
**Goal:** Remove adapter layer - sinks now implement batch interface directly
**Blocked by:** WP-03 ✅

**Rationale:** WP-03 made sinks batch-aware with ArtifactDescriptor returns. SinkAdapter and SinkLike are now redundant indirection layers.

#### Tasks
- [ ] Task 1: Delete `adapters.py` and `test_adapters.py`
- [ ] Task 2: Delete `SinkLike` from `executors.py`
- [ ] Task 3: Update `orchestrator.py` to use `SinkProtocol`
- [ ] Task 4: Update CLI to use sinks directly
- [ ] Task 5: Remove `SinkAdapter` from `engine/__init__.py` exports
- [ ] Task 6: Run full verification

#### Verification
- [ ] `adapters.py` deleted
- [ ] `test_adapters.py` deleted
- [ ] No `SinkLike` anywhere in codebase
- [ ] No `SinkAdapter` anywhere in codebase
- [ ] CLI creates sinks directly (no wrapper)
- [ ] Orchestrator uses `SinkProtocol` type hints
- [ ] All tests pass

---

### WP-04a: Delete *Like Protocol Duplications

**Status:** 🔴 Not Started
**Goal:** Remove TransformLike, GateLike, AggregationLike protocols and rename union alias

**Rationale:** These protocols in executors.py duplicate the full protocols and serve no purpose. Per No Legacy Code Policy, delete them.

**Files:**
- `src/elspeth/engine/executors.py`
- `src/elspeth/engine/orchestrator.py`

#### Tasks
- [ ] Task 1: Delete `TransformLike` protocol from executors.py (~lines 75-83)
- [ ] Task 2: Delete `GateLike` protocol from executors.py (~lines 212-220)
- [ ] Task 3: Delete `AggregationLike` protocol from executors.py (~lines 444-465)
- [ ] Task 4: Update executor methods to use full protocols (TransformProtocol, GateProtocol, AggregationProtocol)
- [ ] Task 5: Rename `TransformLike` union alias to `RowPlugin` in orchestrator.py
- [ ] Task 6: Update all references in orchestrator.py to use `RowPlugin`
- [ ] Task 7: Run mypy and tests

#### Verification
- [ ] No `TransformLike` protocol in executors.py
- [ ] No `GateLike` in executors.py
- [ ] No `AggregationLike` in executors.py
- [ ] Executors use full protocols from `elspeth.plugins.protocols`
- [ ] orchestrator.py uses `RowPlugin` for union alias
- [ ] `mypy --strict` passes
- [ ] All tests pass

---

### WP-05: Audit Schema Enhancement

**Status:** 🔴 Not Started
**Plan:** [2026-01-17-wp05-audit-schema-enhancement.md](./2026-01-17-wp05-audit-schema-enhancement.md)
**Goal:** Add missing columns and fix types for audit completeness
**Unlocks:** WP-06

#### Tasks
- [ ] Task 1: Add TriggerType enum
- [ ] Task 2: Add idempotency_key to artifacts table
- [ ] Task 3: Add trigger_type to batches table
- [ ] Task 4: Fix Batch.status type from str to BatchStatus
- [ ] Task 5: Generate Alembic migration
- [ ] Task 6: Run full verification

#### Verification
- [ ] `TriggerType` enum exists with 5 values
- [ ] `artifacts_table` has `idempotency_key` column
- [ ] `batches_table` has `trigger_type` column
- [ ] `Batch.status` type is `BatchStatus`
- [ ] Alembic migration generated
- [ ] All tests pass

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
**Plan:** [2026-01-17-wp11-orphaned-code-cleanup.md](./2026-01-17-wp11-orphaned-code-cleanup.md)
**Goal:** Remove dead code, KEEP audit-critical infrastructure

#### Decisions Made
- **RetryManager:** KEEP & INTEGRATE (Phase 5)
- **Call infrastructure:** KEEP (Phase 6)
- **on_register():** DELETE (never called)

#### Tasks
- [ ] Task 1: Remove on_register() from 4 base classes
- [ ] Task 2: Verify RetryManager is ready for integration
- [ ] Task 3: Verify Call infrastructure is intact
- [ ] Task 4: Run full verification

#### Verification
- [ ] `on_register()` removed from BaseSource, BaseTransform, BaseGate, BaseAggregation
- [ ] No code calls `on_register()` anywhere
- [ ] RetryManager tests pass
- [ ] Call infrastructure intact
- [ ] All tests pass

---

### WP-12: Utility Consolidation

**Status:** 🔴 Not Started
**Plan:** [2026-01-17-wp12-utility-consolidation.md](./2026-01-17-wp12-utility-consolidation.md)
**Goal:** Extract duplicated code to shared utilities
**Recommended after:** WP-02

#### Tasks
- [ ] Task 1: Create utils.py with get_nested_field()
- [ ] Task 2: Add DynamicSchema class
- [ ] Task 3: Update field_mapper.py to use get_nested_field
- [ ] Task 4: Update sinks to use DynamicSchema
- [ ] Task 5: Run full verification

#### Verification
- [ ] `get_nested_field()` has 9 passing tests
- [ ] `DynamicSchema` has 4 passing tests
- [ ] `field_mapper.py` imports from utils, no local `_get_nested`
- [ ] All sinks use `DynamicSchema` instead of local schema classes
- [ ] All plugin tests pass

---

### WP-13: Sink Test Rewrites

**Status:** 🔴 Not Started
**Goal:** All sink tests use batch signature
**Blocked by:** WP-03, WP-04

**Note:** `test_adapters.py` is deleted in WP-04, so no adapter tests to update.

#### Tasks
- [ ] Rewrite `tests/plugins/sinks/test_csv_sink.py`
- [ ] Rewrite `tests/plugins/sinks/test_json_sink.py`
- [ ] Rewrite `tests/plugins/sinks/test_database_sink.py`
- [ ] Create MockSink fixture for engine tests that need it

#### Verification
- [ ] All sink plugin tests pass
- [ ] No per-row write patterns remain
- [ ] Engine tests use inline MockSink or fixture

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
| | | | |
