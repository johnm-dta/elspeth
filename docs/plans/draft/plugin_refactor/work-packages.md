# Plugin Refactor Work Packages

> **Date:** 2026-01-17
> **Source:** gap-analysis.md, cleanup-list.md
> **Principle:** Minimize seams by grouping changes that touch the same files

## Grouping Strategy

Work packages are organized to:
1. **Touch each file once** - avoid re-opening files across packages
2. **Maintain working state** - each package ends with passing tests
3. **Minimize integration points** - dependencies flow one direction
4. **Enable parallel work** - independent packages can run concurrently

---

## Dependency Graph

```
WP-01 ──┬──► WP-03 ──► WP-04 ──► WP-13
        │
WP-02   │   (independent)
        │
WP-05 ──┴──► WP-06

WP-07 ──┬──► WP-08
        └──► WP-10

WP-09       (independent, large)

WP-11       (independent)
WP-12       (independent)

WP-14       (depends on most above)
```

---

## WP-01: Protocol & Base Class Alignment

**Goal:** Single pass through protocols.py and base.py for all contract changes

**Files:**
- `src/elspeth/plugins/protocols.py`
- `src/elspeth/plugins/base.py`

**Changes:**

| Item | File | Lines |
|------|------|-------|
| Add `determinism` to SourceProtocol | protocols.py | 52-54 |
| Add `plugin_version` to SourceProtocol | protocols.py | 52-54 |
| Add `determinism` to BaseSource | base.py | 302-353 |
| Add `plugin_version` to BaseSource | base.py | 302-353 |
| Change `SinkProtocol.write()` signature | protocols.py | 480-490 |
| Change `BaseSink.write()` signature | base.py | 272-278 |
| Ensure lifecycle hooks exist on all bases | base.py | various |

**Verification:**
- [ ] All protocol attributes match contract
- [ ] Mypy passes on plugin module

**Effort:** Low (~2 hours)
**Dependencies:** None
**Unlocks:** WP-03

---

## WP-02: Gate Plugin Deletion

**Goal:** Complete removal of plugin-based gates (engine gates come later)

**Files to DELETE:**
```
src/elspeth/plugins/gates/filter_gate.py      (249 lines)
src/elspeth/plugins/gates/field_match_gate.py (193 lines)
src/elspeth/plugins/gates/threshold_gate.py   (144 lines)
src/elspeth/plugins/gates/hookimpl.py         (22 lines)
src/elspeth/plugins/gates/__init__.py         (11 lines)
tests/plugins/gates/test_filter_gate.py       (276 lines)
tests/plugins/gates/test_field_match_gate.py  (230 lines)
tests/plugins/gates/test_threshold_gate.py    (221 lines)
tests/plugins/gates/__init__.py               (1 line)
```

**Files to MODIFY:**
| File | Change |
|------|--------|
| `src/elspeth/cli.py` | Remove gate imports (line 228) and registry (241-245) |
| `src/elspeth/plugins/manager.py` | Remove builtin_gates import (161) and registration (168) |
| `tests/plugins/test_base.py` | Remove ThresholdGate tests (74-100) |
| `tests/plugins/test_protocols.py` | Remove ThresholdGate conformance (145-191) |

**What to KEEP:**
- `BaseGate` in base.py (isinstance checks)
- `GateProtocol` in protocols.py (type contract)
- `GateResult`, `RoutingAction` (engine uses these)

**Verification:**
- [ ] No imports of deleted gate plugins anywhere
- [ ] `grep -r "FilterGate\|FieldMatchGate\|ThresholdGate" src/` returns nothing
- [ ] Tests pass (gate tests deleted)

**Effort:** Low (~1 hour)
**Dependencies:** None
**Unlocks:** Nothing (pure cleanup)

---

## WP-03: Sink Implementation Rewrite

**Goal:** All sinks conform to batch signature with ArtifactDescriptor return

**Files:**
- `src/elspeth/plugins/sinks/csv_sink.py`
- `src/elspeth/plugins/sinks/json_sink.py`
- `src/elspeth/plugins/sinks/database_sink.py`

**Changes per sink:**

| Current | New |
|---------|-----|
| `write(row: dict) -> None` | `write(rows: list[dict]) -> ArtifactDescriptor` |
| Per-row lazy initialization | Batch processing |
| No return value | Return ArtifactDescriptor with content_hash, size_bytes |
| Internal buffering | Batch input from engine |

**Content Hashing:**
- CSVSink: SHA-256 of written file
- JSONSink: SHA-256 of written file
- DatabaseSink: SHA-256 of canonical JSON payload before INSERT

**Also add to each sink:**
- `determinism = Determinism.IO_WRITE`
- `plugin_version = "1.0.0"`
- `on_start()` and `on_complete()` lifecycle hooks (even if `pass`)

**Verification:**
- [ ] All sinks return ArtifactDescriptor
- [ ] content_hash is non-empty SHA-256
- [ ] size_bytes > 0 for non-empty writes
- [ ] Mypy passes

**Effort:** Medium (~4 hours)
**Dependencies:** WP-01
**Unlocks:** WP-04, WP-13

---

## WP-04: Sink Adapter Update

**Goal:** SinkAdapter delegates to batch write, removes per-row loop

**Files:**
- `src/elspeth/engine/adapters.py`

**Changes:**

| Current (lines 183-185) | New |
|-------------------------|-----|
| Loop calling `sink.write(row)` per row | Direct `sink.write(rows)` call |
| `_rows_written` counter | Removed (artifact tracks this) |
| No return from write | Capture ArtifactDescriptor |

**Verification:**
- [ ] `SinkAdapter.write()` returns ArtifactDescriptor
- [ ] No per-row iteration in adapter
- [ ] Integration with orchestrator works

**Effort:** Low (~1 hour)
**Dependencies:** WP-03
**Unlocks:** WP-13

---

## WP-05: Audit Schema Enhancement

**Goal:** Add missing columns and fix types for audit completeness

**Files:**
- `src/elspeth/core/landscape/schema.py`
- `src/elspeth/core/landscape/models.py`
- `src/elspeth/contracts/enums.py`
- `src/elspeth/contracts/audit.py`

**Schema Changes:**

| Table | Column | Type | Purpose |
|-------|--------|------|---------|
| `artifacts` | `idempotency_key` | `String(256)` | Retry deduplication |
| `batches` | `trigger_type` | `String(32)` | Typed trigger enum |

**New Enum:**
```python
class TriggerType(str, Enum):
    COUNT = "count"
    TIMEOUT = "timeout"
    CONDITION = "condition"
    END_OF_SOURCE = "end_of_source"
    MANUAL = "manual"
```

**Model Fixes:**
| File | Field | Current | Fixed |
|------|-------|---------|-------|
| models.py:268 | `Batch.status` | `str` | `BatchStatus` |

**Verification:**
- [ ] Alembic migration generated
- [ ] Models match schema
- [ ] Mypy passes on contracts

**Effort:** Medium (~2 hours)
**Dependencies:** None
**Unlocks:** WP-06

---

## WP-06: Aggregation Triggers

**Goal:** Config-driven aggregation triggers replace plugin-driven decisions

**Files:**
- `src/elspeth/core/config.py` (new AggregationSettings)
- `src/elspeth/engine/orchestrator.py`
- `src/elspeth/engine/executors.py` (AggregationExecutor)

**New Config:**
```python
class AggregationSettings(BaseModel):
    plugin: str
    trigger: TriggerConfig  # count, timeout, condition
    output_mode: Literal["single", "passthrough", "transform"]
```

**Engine Changes:**
- Orchestrator evaluates trigger conditions
- AggregationExecutor.accept() only accepts/rejects
- Trigger decision moves from plugin to engine

**Stale Code After:**
- `AcceptResult.trigger` field (still generated, not read)
- `BaseAggregation.should_trigger()` (defined, never called)

**Verification:**
- [ ] Config validation rejects invalid triggers
- [ ] All 4 trigger types work: count, timeout, condition, end_of_source
- [ ] Output modes work: single, passthrough, transform

**Effort:** Medium-High (~6 hours)
**Dependencies:** WP-05
**Unlocks:** WP-14 (partial)

---

## WP-07: Fork Work Queue

**Goal:** Forked child tokens actually execute through their paths

**Files:**
- `src/elspeth/engine/processor.py`

**Changes:**

| Current (line 91) | New |
|-------------------|-----|
| "LINEAR pipelines only" | Work queue processes fork children |
| Returns FORKED, children orphaned | Children queued and processed |
| Single-pass execution | Loop until queue empty |

**Implementation:**
```python
def process_row(...):
    work_queue = deque([initial_token])
    results = []

    while work_queue:
        token = work_queue.popleft()
        result = self._process_single_token(token, ...)

        if result.outcome == RowOutcome.FORKED:
            work_queue.extend(result.child_tokens)
        else:
            results.append(result)

    return results
```

**Verification:**
- [ ] Fork creates children that execute
- [ ] Each child follows its assigned path
- [ ] Parent FORKED, children reach terminal states
- [ ] Audit trail shows complete lineage

**Effort:** High (~8 hours)
**Dependencies:** None
**Unlocks:** WP-08, WP-10

---

## WP-08: Coalesce Executor

**Goal:** Merge tokens from parallel fork paths

**Files:**
- `src/elspeth/engine/coalesce_executor.py` (NEW)
- `src/elspeth/engine/processor.py` (add coalesce handling)
- `src/elspeth/engine/__init__.py` (export)

**Implementation:**

| Component | Status | Action |
|-----------|--------|--------|
| CoalesceProtocol | Exists | Use as-is |
| CoalescePolicy enum | Exists | Use as-is |
| LandscapeRecorder.coalesce_tokens() | Exists | Call from executor |
| CoalesceExecutor | Missing | CREATE |
| Policy enforcement | Missing | IMPLEMENT |
| Merge strategies | Missing | IMPLEMENT |

**Policies to implement:**
- `require_all` - Wait for all branches
- `quorum` - Wait for N branches
- `best_effort` - Merge whatever arrives
- `first` - Take first arrival

**Merge strategies:**
- `union` - Combine all fields
- `nested` - Each branch as nested object
- `select` - Take specific branch output

**Verification:**
- [ ] COALESCED terminal state reachable
- [ ] All 4 policies work
- [ ] All 3 merge strategies work
- [ ] Timeout handling works

**Effort:** High (~8 hours)
**Dependencies:** WP-07
**Unlocks:** WP-14 (partial)

---

## WP-09: Engine-Level Gates

**Goal:** Gates become config-driven engine operations with safe expression parsing

**Files:**
- `src/elspeth/engine/expression_parser.py` (NEW)
- `src/elspeth/core/config.py` (add GateSettings)
- `src/elspeth/engine/orchestrator.py` (route resolution refactor)
- `src/elspeth/engine/executors.py` (simplify GateExecutor)

**Expression Parser:**
```python
# Safe evaluation - NOT Python eval()
allowed = {
    "field_access": ["row['field']", "row.get('field')"],
    "comparisons": ["==", "!=", "<", ">", "<=", ">="],
    "boolean": ["and", "or", "not"],
    "membership": ["in", "not in"],
    "literals": [strings, numbers, booleans, None],
}
```

**GateSettings Config:**
```yaml
gates:
  - name: quality_check
    condition: "row['confidence'] >= 0.85"
    routes:
      high: continue
      low: review_sink
    fork_to:  # Optional
      - path_a
      - path_b
```

**Route Resolution:**
- Move from GateExecutor to Orchestrator
- Pre-compute at pipeline construction
- Executor just evaluates condition, returns route label

**Verification:**
- [ ] Expression parser rejects unsafe code
- [ ] Composite conditions work: `row['a'] > 0 and row['b'] == 'x'`
- [ ] fork_to creates child tokens
- [ ] Route labels resolve correctly

**Effort:** High (~10 hours)
**Dependencies:** None (but should come after WP-02)
**Unlocks:** WP-14 (partial)

---

## WP-10: Quarantine Implementation

**Goal:** QUARANTINED terminal state becomes reachable

**Files:**
- `src/elspeth/engine/processor.py`

**Implementation:**
- Add quarantine logic for malformed/invalid rows
- Source validation layer
- Record quarantine reason in audit trail

**When to quarantine:**
- Row fails schema validation
- Required fields missing
- Type coercion fails
- External validation fails

**Verification:**
- [ ] QUARANTINED state reachable
- [ ] Quarantine reason recorded
- [ ] Pipeline continues after quarantine (doesn't crash)

**Effort:** Medium (~4 hours)
**Dependencies:** WP-07 (touches same file)
**Unlocks:** WP-14 (partial)

---

## WP-11: Orphaned Code Cleanup

**Goal:** Remove dead code that was never integrated

**Files:**

| File | Lines | Item | Action |
|------|-------|------|--------|
| `engine/retry.py` | 37-156 | RetryManager | DELETE or integrate |
| `contracts/enums.py` | 144-147 | CallType | DELETE |
| `contracts/enums.py` | 156-157 | CallStatus | DELETE |
| `contracts/audit.py` | 237-252 | Call dataclass | DELETE |
| `landscape/recorder.py` | 1707-1743 | get_calls() | DELETE |
| `plugins/base.py` | 210-213 | should_trigger() | Mark deprecated |
| `plugins/base.py` | 219-223 | reset() | Mark deprecated |
| `plugins/base.py` | various | on_register() | DELETE (never called) |

**Decision needed:**
- RetryManager: DELETE (unused) or INTEGRATE (useful for Phase 5)?
- Call infrastructure: DELETE (Phase 6 feature, rebuild when needed)?

**Verification:**
- [ ] No references to deleted code
- [ ] Tests pass
- [ ] No import errors

**Effort:** Low (~2 hours)
**Dependencies:** None
**Unlocks:** Nothing (pure cleanup)

---

## WP-12: Utility Consolidation

**Goal:** Extract duplicated code to shared utilities

**Files:**
- `src/elspeth/plugins/utils.py` (NEW)
- `src/elspeth/plugins/transforms/field_mapper.py`

**Duplicated Code:**

`_get_nested()` - Extract to utils.py:
```python
def get_nested_field(data: dict, path: str, default: Any = MISSING) -> Any:
    """Traverse nested dict using dot notation path."""
    parts = path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
```

**Optional:** DynamicPluginSchema factory
```python
def create_dynamic_schema(name: str) -> type[PluginSchema]:
    """Create a schema that accepts any fields."""
    return type(name, (PluginSchema,), {"model_config": {"extra": "allow"}})
```

**Update field_mapper.py:**
- Import from utils instead of defining locally
- (Gate files already deleted in WP-02)

**Verification:**
- [ ] `_get_nested` exists in only one location
- [ ] field_mapper.py works with imported utility

**Effort:** Low (~1 hour)
**Dependencies:** None (but after WP-02)
**Unlocks:** Nothing (pure cleanup)

---

## WP-13: Sink Test Rewrites

**Goal:** All sink tests use batch signature

**Files:**
- `tests/plugins/sinks/test_csv_sink.py`
- `tests/plugins/sinks/test_json_sink.py`
- `tests/plugins/sinks/test_database_sink.py`
- `tests/engine/test_adapters.py` (MockSink class)

**Test Pattern Change:**

```python
# OLD (per-row)
sink.write({"id": "1"}, ctx)
sink.write({"id": "2"}, ctx)

# NEW (batch)
artifact = sink.write([{"id": "1"}, {"id": "2"}], ctx)
assert isinstance(artifact, ArtifactDescriptor)
assert artifact.content_hash  # non-empty
assert artifact.size_bytes > 0
```

**MockSink Update:**
```python
class MockSink:
    def write(self, rows: list[dict], ctx) -> ArtifactDescriptor:
        self.rows_written.extend(rows)
        return ArtifactDescriptor.for_file(...)
```

**Verification:**
- [ ] All sink tests pass
- [ ] Adapter tests pass
- [ ] No per-row write patterns remain

**Effort:** Medium (~4 hours)
**Dependencies:** WP-03, WP-04
**Unlocks:** Nothing (verification)

---

## WP-14: Engine Test Rewrites

**Goal:** Engine tests updated for all architectural changes

**Files:**
- `tests/engine/test_processor.py` (828 lines)
- `tests/engine/test_executors.py` (1956 lines)
- `tests/engine/test_orchestrator.py` (3920+ lines)
- `tests/engine/test_integration.py` (1048 lines)
- `tests/plugins/test_integration.py` (237 lines)

**Changes per file:**

| File | Changes |
|------|---------|
| test_processor.py | Fork work queue, coalesce, quarantine |
| test_executors.py | Aggregation triggers, gate routing |
| test_orchestrator.py | Engine gates, route resolution |
| test_integration.py | End-to-end with new architecture |

**Estimated test count:** ~450 tests affected

**Verification:**
- [ ] All tests pass
- [ ] Coverage maintained
- [ ] No references to old patterns

**Effort:** High (~16+ hours)
**Dependencies:** WP-06, WP-07, WP-08, WP-09, WP-10
**Unlocks:** Nothing (final verification)

---

## Execution Order

### Critical Path (Sequential)

```
WP-01 → WP-03 → WP-04 → WP-13
```

### Parallel Tracks

**Track A: Sink Contract**
```
WP-01 → WP-03 → WP-04 → WP-13
```

**Track B: DAG Execution**
```
WP-07 → WP-08
      → WP-10
```

**Track C: Aggregation**
```
WP-05 → WP-06
```

**Track D: Engine Gates**
```
WP-09
```

**Track E: Cleanup (Anytime)**
```
WP-02 (early)
WP-11 (anytime)
WP-12 (after WP-02)
```

**Final:**
```
WP-14 (after all others)
```

---

## Suggested Sprint Allocation

### Sprint 1: Foundation & Cleanup
- WP-01: Protocol & Base Class Alignment
- WP-02: Gate Plugin Deletion
- WP-05: Audit Schema Enhancement
- WP-11: Orphaned Code Cleanup

### Sprint 2: Sink Contract
- WP-03: Sink Implementation Rewrite
- WP-04: Sink Adapter Update
- WP-12: Utility Consolidation
- WP-13: Sink Test Rewrites

### Sprint 3: DAG & Aggregation
- WP-06: Aggregation Triggers
- WP-07: Fork Work Queue
- WP-10: Quarantine Implementation

### Sprint 4: Advanced Features
- WP-08: Coalesce Executor
- WP-09: Engine-Level Gates

### Sprint 5: Verification
- WP-14: Engine Test Rewrites
- Final integration testing

---

## Effort Summary

| WP | Effort | Hours |
|----|--------|-------|
| WP-01 | Low | 2 |
| WP-02 | Low | 1 |
| WP-03 | Medium | 4 |
| WP-04 | Low | 1 |
| WP-05 | Medium | 2 |
| WP-06 | Medium-High | 6 |
| WP-07 | High | 8 |
| WP-08 | High | 8 |
| WP-09 | High | 10 |
| WP-10 | Medium | 4 |
| WP-11 | Low | 2 |
| WP-12 | Low | 1 |
| WP-13 | Medium | 4 |
| WP-14 | High | 16 |

**Total: ~69 hours** (not counting parallel execution)

---

## Risk Matrix

| WP | Risk | Mitigation |
|----|------|------------|
| WP-03 | Content hashing edge cases | Test with large files, binary data |
| WP-07 | Infinite loops in work queue | Max iteration guard |
| WP-08 | Timeout race conditions | Use monotonic clock |
| WP-09 | Expression parser security | Extensive fuzzing |
| WP-14 | Large test rewrite scope | Incremental, focus on critical paths |
