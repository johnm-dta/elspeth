# Plugin Refactor Cleanup List

> **Date:** 2026-01-17
> **Companion to:** gap-analysis.md
> **Purpose:** Identify dead code, obsolete implementations, and cleanup tasks

## Executive Summary

This document catalogs code that will become dead, obsolete, or require cleanup as part of the plugin protocol refactor. The analysis identified:

| Category | Files | Lines | Action |
|----------|-------|-------|--------|
| Gate plugin code | 9 | 1,347 | DELETE |
| Sink signature changes | 6 | ~200 | REWRITE |
| Orphaned implementations | 8 | ~300 | DELETE/FIX |
| Duplicate code | 4 | ~80 | CONSOLIDATE |
| Superseded code | 5 | ~400 | REWRITE |
| Obsolete tests | 14 | ~9,000 | DELETE/REWRITE |
| Schema/config changes | 6 | ~100 | MODIFY |

**Total cleanup scope:** ~11,500 lines across ~50 files

---

## 1. Gate Plugin Code (DELETE ENTIRELY)

Gates are moving from plugin-level to engine-level. All plugin gate code becomes dead.

### Complete File Deletions

| File | Lines | Content |
|------|-------|---------|
| `src/elspeth/plugins/gates/filter_gate.py` | 249 | FilterGate class + config |
| `src/elspeth/plugins/gates/field_match_gate.py` | 193 | FieldMatchGate class + config |
| `src/elspeth/plugins/gates/threshold_gate.py` | 144 | ThresholdGate class + config |
| `src/elspeth/plugins/gates/hookimpl.py` | 22 | Plugin registration hook |
| `src/elspeth/plugins/gates/__init__.py` | 11 | Public exports |
| `tests/plugins/gates/test_filter_gate.py` | 276 | FilterGate tests |
| `tests/plugins/gates/test_field_match_gate.py` | 230 | FieldMatchGate tests |
| `tests/plugins/gates/test_threshold_gate.py` | 221 | ThresholdGate tests |
| `tests/plugins/gates/__init__.py` | 1 | Package marker |

**Total: 1,347 lines to delete**

### Partial File Deletions

| File | Lines | Content to Remove |
|------|-------|-------------------|
| `src/elspeth/cli.py` | 228, 241-245 | Gate plugin imports and registry |
| `src/elspeth/plugins/manager.py` | 161, 168 | builtin_gates import and registration |
| `src/elspeth/plugins/hookspecs.py` | 66-72 | `elspeth_get_gates` hookspec (optional) |

### What to KEEP

- `BaseGate` class in `base.py` - Used for isinstance() checks
- `GateProtocol` in `protocols.py` - Type contract
- `GateResult` and `RoutingAction` - Engine still uses these

---

## 2. Sink Write Signature Changes (REWRITE)

Sinks changing from `write(row) -> None` to `write(rows) -> ArtifactDescriptor`.

### Implementation Files to Rewrite

| File | Lines | Current | New |
|------|-------|---------|-----|
| `src/elspeth/plugins/sinks/csv_sink.py` | 53-69 | Per-row write + lazy init | Batch write |
| `src/elspeth/plugins/sinks/json_sink.py` | 59-73 | Per-row write + format routing | Batch write |
| `src/elspeth/plugins/sinks/database_sink.py` | 80-99 | Per-row buffering + flush | Batch write |
| `src/elspeth/plugins/base.py` | 272-278 | `write(row) -> None` | `write(rows) -> ArtifactDescriptor` |
| `src/elspeth/plugins/protocols.py` | 480-490 | Per-row signature | Batch signature |

### Adapter Code to Modify

| File | Lines | Change |
|------|-------|--------|
| `src/elspeth/engine/adapters.py` | 183-185 | Remove per-row loop, direct delegation |
| `src/elspeth/engine/adapters.py` | 139-140 | Remove `_rows_written` tracking |

### Test Files to Rewrite

| File | Lines | Reason |
|------|-------|--------|
| `tests/plugins/sinks/test_csv_sink.py` | 34-113 | All per-row write tests |
| `tests/plugins/sinks/test_json_sink.py` | 34-128 | All per-row write tests |
| `tests/plugins/sinks/test_database_sink.py` | 39-82 | All per-row write tests |
| `tests/engine/test_adapters.py` | 11-28 | MockSink per-row signature |

---

## 3. Orphaned/Unused Code (DELETE)

Code that exists but is never called in production.

### Critical: Never-Used Implementations

| File | Lines | Item | Issue |
|------|-------|------|-------|
| `src/elspeth/engine/retry.py` | 37-156 | `RetryManager`, `RetryConfig` | Fully implemented, never instantiated |
| `src/elspeth/contracts/enums.py` | 144-147 | `CallType` enum | LLM, HTTP, SQL, FILESYSTEM - never used |
| `src/elspeth/contracts/enums.py` | 156-157 | `CallStatus` enum | SUCCESS, ERROR - never used |
| `src/elspeth/contracts/audit.py` | 237-252 | `Call` dataclass | Defined, never created |
| `src/elspeth/core/landscape/recorder.py` | 1707-1743 | `get_calls()` | Reads from never-populated table |

### Medium: Lifecycle Hooks Never Called

| File | Lines | Method | Issue |
|------|-------|--------|-------|
| `src/elspeth/plugins/base.py` | 210-213 | `BaseAggregation.should_trigger()` | Abstract, never called by engine |
| `src/elspeth/plugins/base.py` | 219-223 | `BaseAggregation.reset()` | Optional, never called |
| `src/elspeth/plugins/base.py` | multiple | `on_register()` on all bases | Never called by orchestrator |

### Low: Backward Compatibility Properties

| File | Lines | Item | Issue |
|------|-------|------|-------|
| `src/elspeth/contracts/results.py` | 103-112 | `RowResult.token_id`, `.row_id` | Duplicate of `result.token.token_id` |

---

## 4. Duplicate/Redundant Code (CONSOLIDATE)

### Critical: `_get_nested()` Duplicated 4x

**Identical implementation in 4 files:**

| File | Lines |
|------|-------|
| `src/elspeth/plugins/gates/filter_gate.py` | 227-245 |
| `src/elspeth/plugins/gates/threshold_gate.py` | 122-140 |
| `src/elspeth/plugins/gates/field_match_gate.py` | 171-189 |
| `src/elspeth/plugins/transforms/field_mapper.py` | 90-108 |

**Action:** Extract to `src/elspeth/plugins/utils.py` as shared utility.

**Note:** Gate files will be deleted anyway (see Section 1), but `field_mapper.py` should use the shared utility.

### Medium: Dynamic Schema Class Pattern (14+ occurrences)

Every source/sink/transform creates identical pattern:

```python
class XXXSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""
    model_config = {"extra": "allow"}
```

**Action:** Create `DynamicPluginSchema` factory or base class.

### Low: Empty `close()` Methods (6 locations)

All return `pass` with same docstring. Already in files marked for deletion (gates) or acceptable (transforms).

---

## 5. Code Superseded by Gap Fixes (REWRITE)

Code that exists but will be substantially rewritten.

### Aggregation Trigger Logic

| File | Lines | Current | After Fix |
|------|-------|---------|-----------|
| `src/elspeth/engine/processor.py` | 146-167 | Plugin decides trigger via `accept_result.trigger` | Engine evaluates trigger config |
| `src/elspeth/engine/executors.py` | 506-600 | `AggregationExecutor.accept()` checks trigger | Accept only, engine triggers |

**Stale after fix:**
- `AcceptResult.trigger` field - generated but not read
- `BaseAggregation.should_trigger()` - defined but not called

### Gate Route Resolution

| File | Lines | Current | After Fix |
|------|-------|---------|-----------|
| `src/elspeth/engine/executors.py` | 330-379 | Route resolution in executor | Route resolution in orchestrator |
| `src/elspeth/engine/executors.py` | 405-441 | `_record_routing()` with route map | Simplified audit recording |

### Token Work Queue (Fork/Join)

| File | Lines | Current | After Fix |
|------|-------|---------|-----------|
| `src/elspeth/engine/processor.py` | 81-195 | Linear only, returns FORKED | Work queue processes children |

**Comment at line 91:**
> "NOTE: This implementation handles LINEAR pipelines only. For DAG support (fork/join), this needs a work queue..."

### Node State Recording

| File | Lines | Change |
|------|-------|--------|
| `src/elspeth/core/landscape/recorder.py` | 842-900 | Add `idempotency_key` parameter |
| `src/elspeth/core/landscape/recorder.py` | 902-969 | Add `lifecycle_event` parameter |

---

## 6. Schema and Config Changes (MODIFY)

### Missing Configuration Classes

| Item | Location | Action |
|------|----------|--------|
| `AggregationSettings` | `src/elspeth/core/config.py` | CREATE - trigger config, output modes |
| `GateSettings` | `src/elspeth/core/config.py` | CREATE - when gates become config-driven |

### Schema Columns to Add

| Table | Column | Type | Purpose |
|-------|--------|------|---------|
| `artifacts` | `idempotency_key` | `String(256)` | Retry deduplication |
| `batches` | `trigger_type` | `String(32)` | Typed trigger enum |

**Files:**
- `src/elspeth/core/landscape/schema.py` - Add columns
- `src/elspeth/core/landscape/models.py` - Add fields to dataclasses
- `src/elspeth/contracts/enums.py` - Add `TriggerType` enum

### Model Type Mismatches

| File | Line | Field | Current | Should Be |
|------|------|-------|---------|-----------|
| `src/elspeth/core/landscape/models.py` | 268 | `Batch.status` | `str` | `BatchStatus` |

---

## 7. Test Code Obsolescence (DELETE/REWRITE)

### Complete Test File Deletions

| File | Lines | Reason |
|------|-------|--------|
| `tests/plugins/gates/test_filter_gate.py` | 276 | Gate becomes engine-level |
| `tests/plugins/gates/test_field_match_gate.py` | 230 | Gate becomes engine-level |
| `tests/plugins/gates/test_threshold_gate.py` | 221 | Gate becomes engine-level |

**Delete entire directory:** `tests/plugins/gates/`

### Major Test File Rewrites

| File | Lines | Tests Affected | Reason |
|------|-------|----------------|--------|
| `tests/plugins/sinks/test_csv_sink.py` | 115 | ~15 | Batch signature |
| `tests/plugins/sinks/test_json_sink.py` | 128 | ~18 | Batch signature |
| `tests/plugins/sinks/test_database_sink.py` | 103 | ~12 | Batch signature |
| `tests/plugins/test_integration.py` | 237 | ~5 | Gate plugin usage |
| `tests/engine/test_adapters.py` | 362 | ~25 | MockSink signature |
| `tests/engine/test_integration.py` | 1048 | ~50 | Sink/gate semantics |
| `tests/engine/test_processor.py` | 828 | ~60 | Batch sink semantics |
| `tests/engine/test_orchestrator.py` | 3920+ | ~200 | Gate/sink changes |
| `tests/engine/test_executors.py` | 1956 | ~60 | Gate executor pattern |

### Partial Test Deletions

| File | Lines | Content |
|------|-------|---------|
| `tests/plugins/test_base.py` | 74-100 | ThresholdGate test class |
| `tests/plugins/test_protocols.py` | 145-191 | ThresholdGate conformance |
| `tests/engine/test_plugin_detection.py` | 26-28, 125-127 | Gate plugin detection |

### Test Impact Summary

- **Tests to delete:** ~38 (gate plugin tests)
- **Tests to rewrite:** ~450+ (sink/executor/orchestrator)
- **Total lines affected:** ~9,000 (32% of test codebase)

---

## 8. Documentation Updates

### Docstring Examples to Update

| File | Lines | Current | New |
|------|-------|---------|-----|
| `src/elspeth/plugins/base.py` | 248-249 | `write(row) -> None` | `write(rows) -> ArtifactDescriptor` |
| `src/elspeth/plugins/protocols.py` | 460-461 | `write(row) -> None` | `write(rows) -> ArtifactDescriptor` |

---

## 9. Cleanup Priority Order

### Phase 1: Quick Wins (Low Risk)

1. Delete `tests/plugins/gates/` directory (727 lines)
2. Delete gate plugin files (619 lines)
3. Remove gate imports from `cli.py` and `manager.py`
4. Extract `_get_nested()` to shared utility

### Phase 2: Sink Signature (Medium Risk)

1. Update `BaseSink.write()` signature in `base.py`
2. Update `SinkProtocol.write()` signature in `protocols.py`
3. Rewrite all 3 sink implementations
4. Update `SinkAdapter` in `adapters.py`
5. Rewrite sink tests

### Phase 3: Schema Changes (Medium Risk)

1. Add `idempotency_key` to artifacts table
2. Add `trigger_type` to batches table
3. Update model dataclasses
4. Add `TriggerType` enum

### Phase 4: Engine Changes (High Risk)

1. Aggregation trigger refactor
2. Gate route resolution move
3. Token work queue implementation
4. Coalesce executor creation

### Phase 5: Large Test Rewrites

1. `test_orchestrator.py` (3920+ lines)
2. `test_executors.py` (1956 lines)
3. `test_integration.py` (1048 lines)
4. `test_processor.py` (828 lines)

---

## 10. Verification Checklist

After cleanup, verify:

- [ ] No imports of deleted gate plugins
- [ ] No references to `elspeth_get_gates` hook
- [ ] All sinks return `ArtifactDescriptor` with `content_hash`
- [ ] All sinks accept `rows: list[dict]` not `row: dict`
- [ ] `_get_nested()` exists in only one location
- [ ] `artifacts` table has `idempotency_key` column
- [ ] `batches` table has `trigger_type` column
- [ ] `Batch.status` uses `BatchStatus` enum
- [ ] All tests pass after rewrites
- [ ] No orphaned imports in `__init__.py` files

---

## Appendix: Files by Action

### DELETE (Complete Files)

```
src/elspeth/plugins/gates/filter_gate.py
src/elspeth/plugins/gates/field_match_gate.py
src/elspeth/plugins/gates/threshold_gate.py
src/elspeth/plugins/gates/hookimpl.py
src/elspeth/plugins/gates/__init__.py
tests/plugins/gates/test_filter_gate.py
tests/plugins/gates/test_field_match_gate.py
tests/plugins/gates/test_threshold_gate.py
tests/plugins/gates/__init__.py
```

### REWRITE (Substantial Changes)

```
src/elspeth/plugins/sinks/csv_sink.py
src/elspeth/plugins/sinks/json_sink.py
src/elspeth/plugins/sinks/database_sink.py
src/elspeth/plugins/base.py (BaseSink.write)
src/elspeth/plugins/protocols.py (SinkProtocol.write)
src/elspeth/engine/adapters.py (SinkAdapter)
src/elspeth/engine/processor.py (work queue)
src/elspeth/engine/executors.py (aggregation triggers, gate routing)
```

### MODIFY (Add Columns/Fields)

```
src/elspeth/core/config.py (add AggregationSettings)
src/elspeth/core/landscape/schema.py (add columns)
src/elspeth/core/landscape/models.py (add fields)
src/elspeth/contracts/enums.py (add TriggerType)
```

### CREATE (New Files)

```
src/elspeth/plugins/utils.py (shared utilities)
src/elspeth/engine/expression_parser.py (gate expressions)
src/elspeth/engine/coalesce_executor.py (coalesce logic)
```
