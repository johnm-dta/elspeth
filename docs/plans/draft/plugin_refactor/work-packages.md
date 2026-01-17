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
WP-01 ──┬──► WP-03 ──► WP-04 ──► WP-04a ──► WP-13
        │
WP-02 ──┼──► WP-09       (back-to-back, no gap between deletion and replacement)
        │
WP-05 ──┴──► WP-06

WP-07 ──┬──► WP-08
        └──► WP-10

WP-11       (independent)
WP-12       (after WP-02)

WP-14       (after WP-06, WP-08, WP-09, WP-10)
```

**Critical constraint:** WP-02 (delete gate plugins) and WP-09 (engine gates) must execute back-to-back. WP-09's engine gate tests must pass before WP-02 deletions are considered complete.

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

**Rollback Trigger:** Any hash mismatch in audit trail verification tests, or any silent data integrity failure.

---

## WP-04: Delete SinkAdapter & SinkLike

**Goal:** Remove the adapter layer entirely - sinks now implement batch interface directly

**Rationale:** WP-03 made sinks batch-aware with ArtifactDescriptor returns. The `SinkAdapter` wrapper and `SinkLike` protocol are now redundant indirection layers that add complexity without value. Per No Legacy Code Policy, delete them completely.

**Files to DELETE:**
- `src/elspeth/engine/adapters.py`
- `tests/engine/test_adapters.py`

**Files to MODIFY:**

| File | Change |
|------|--------|
| `src/elspeth/engine/executors.py` | Delete `SinkLike` protocol (lines 668-692) |
| `src/elspeth/engine/executors.py` | Update `SinkExecutor.write()` to use `SinkProtocol` |
| `src/elspeth/engine/orchestrator.py` | Remove `SinkLike` import, use `SinkProtocol` |
| `src/elspeth/engine/orchestrator.py` | Update `PipelineConfig.sinks` type hint |
| `src/elspeth/engine/__init__.py` | Remove `SinkAdapter` export |
| `src/elspeth/cli.py` | Remove `SinkAdapter` import and usage, use sinks directly |

**Verification:**
- [ ] `adapters.py` deleted
- [ ] `test_adapters.py` deleted
- [ ] No `SinkLike` anywhere in codebase
- [ ] No `SinkAdapter` anywhere in codebase
- [ ] CLI creates sinks directly (no wrapper)
- [ ] Orchestrator uses `SinkProtocol` type hints
- [ ] All tests pass

**Effort:** Medium (~2 hours)
**Dependencies:** WP-03
**Unlocks:** WP-04a, WP-13

---

## WP-04a: Delete Remaining *Like Protocol Duplications

**Goal:** Delete AggregationLike protocol, move batch state to executor internal storage, rename TransformLike union alias

**Prior Progress:**
- ✅ TransformLike protocol - DELETED (commit f08c19a)
- ✅ GateLike protocol - DELETED (commit f08c19a)
- ❌ AggregationLike protocol - Still exists (has `_batch_id` attribute)

**Rationale:** `AggregationLike` exists solely to declare `_batch_id: str | None`, which the executor monkey-patches onto plugins. This is a design smell. **Option C:** Move batch state tracking entirely into `AggregationExecutor._batch_ids` dict, then delete the redundant protocol.

**Files to MODIFY:**

| File | Change |
|------|--------|
| `src/elspeth/engine/executors.py` | Add `_batch_ids: dict[str, str \| None]` to AggregationExecutor |
| `src/elspeth/engine/executors.py` | Replace `aggregation._batch_id` with `self._batch_ids[node_id]` |
| `src/elspeth/engine/executors.py` | Add `get_batch_id(node_id)` helper for test access |
| `src/elspeth/engine/executors.py` | Delete `AggregationLike` protocol (~lines 428-449) |
| `src/elspeth/engine/orchestrator.py` | Rename `TransformLike` alias to `RowPlugin` (lines 29, 45, 182, 224) |
| `tests/engine/test_executors.py` | Update assertions to use `executor.get_batch_id()` |
| `tests/engine/test_processor.py` | Remove `_batch_id` from mock aggregations |

**Verification:**
- [ ] `AggregationExecutor._batch_ids` dict manages batch state
- [ ] No `aggregation._batch_id` references in codebase
- [ ] No `AggregationLike` in executors.py
- [ ] `AggregationExecutor.accept()` uses `AggregationProtocol`
- [ ] orchestrator.py uses `RowPlugin` (not `TransformLike`) for union alias
- [ ] `mypy --strict` passes
- [ ] All tests pass

**Effort:** Medium (~1.5-2 hours)
**Dependencies:** WP-04 (same cleanup pattern, avoids merge conflicts)
**Unlocks:** Nothing (pure cleanup, but enables cleaner WP-06 aggregation work)

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

**Cleanup (don't leave stale code):**
- DELETE `AcceptResult.trigger` field from `contracts/results.py`
- DELETE `BaseAggregation.should_trigger()` from `plugins/base.py`
- DELETE `BaseAggregation.reset()` from `plugins/base.py`
- UPDATE any tests that reference these removed items

**Verification:**
- [ ] Config validation rejects invalid triggers
- [ ] All 4 trigger types work: count, timeout, condition, end_of_source
- [ ] Output modes work: single, passthrough, transform
- [ ] No references to `AcceptResult.trigger` remain
- [ ] No references to `should_trigger()` or `reset()` remain

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
- [ ] Max iteration guard prevents infinite loops

**Effort:** High (~8 hours)
**Dependencies:** None
**Unlocks:** WP-08, WP-10

**Rollback Trigger:** Any test showing token loss (children created but not processed) or infinite loop detection.

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

**Files to CREATE:**
- `src/elspeth/engine/expression_parser.py` (safe expression evaluation)
- `tests/engine/test_expression_parser.py` (unit tests for parser security)
- `tests/engine/test_engine_gates.py` (integration tests for gate routing)

**Files to MODIFY:**
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
- [ ] Composite conditions work: `row['a'] > 0 and row['b'] == 'x'`
- [ ] fork_to creates child tokens
- [ ] Route labels resolve correctly

**Security Verification (MANDATORY):**
- [ ] `__import__('os').system('rm -rf /')` → rejected at parse time
- [ ] `eval('malicious')` → rejected at parse time
- [ ] `exec('code')` → rejected at parse time
- [ ] `lambda: ...` → rejected at parse time
- [ ] `[x for x in ...]` (comprehensions) → rejected at parse time
- [ ] Attribute access beyond `row[...]` and `row.get(...)` → rejected
- [ ] Function calls other than `row.get()` → rejected
- [ ] Assignment expressions (`:=`) → rejected
- [ ] Fuzz test with 1000+ random malformed inputs → no crashes, no code execution

**Effort:** High (~10 hours)
**Dependencies:** None (but should come after WP-02)
**Unlocks:** WP-14 (partial)

**Rollback Trigger:** If engine gates cannot replicate plugin gate behavior for existing test cases, halt and reassess.

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

**Goal:** Remove dead code that was never integrated, KEEP audit-critical infrastructure

> **NOTE:** Split this WP into sub-tasks when execution begins - some items are deletions, others are integrations.

**Files:**

| File | Lines | Item | Action |
|------|-------|------|--------|
| `engine/retry.py` | 37-156 | RetryManager | **KEEP & INTEGRATE** (Phase 5 retry audit) |
| `contracts/enums.py` | 144-147 | CallType | **KEEP** (Phase 6 external call audit) |
| `contracts/enums.py` | 156-157 | CallStatus | **KEEP** (Phase 6 external call audit) |
| `contracts/audit.py` | 237-252 | Call dataclass | **KEEP** (Phase 6 external call audit) |
| `landscape/recorder.py` | 1707-1743 | get_calls() | **KEEP** (Phase 6 external call audit) |
| `plugins/base.py` | various | on_register() | DELETE (never called) |

**Decisions made (2026-01-17):**
- **RetryManager:** KEEP & INTEGRATE - Retries must be auditable with `(run_id, row_id, transform_seq, attempt)`
- **Call infrastructure:** KEEP for Phase 6 - External calls (LLMs, APIs) are a major audit surface

**Items moved to WP-06:**
- `AcceptResult.trigger` field - cleaned up when WP-06 makes it obsolete
- `BaseAggregation.should_trigger()` - cleaned up when WP-06 makes it obsolete
- `BaseAggregation.reset()` - cleaned up when WP-06 makes it obsolete

**Verification:**
- [ ] on_register() removed from base classes
- [ ] RetryManager integrated into engine retry flow
- [ ] Tests pass
- [ ] No import errors

**Effort:** Low (~2 hours)
**Dependencies:** None
**Unlocks:** Nothing (pure cleanup)

---

## WP-11.99: Config-Driven Plugin Schemas

**Goal:** Replace hardcoded `extra="allow"` schemas with mandatory config-driven schema definitions

**Architecture:** Every plugin that processes row data must declare `schema` in config. Two modes:
- `fields: dynamic` - Accept anything (logged for audit)
- Explicit fields with `mode: strict` (exactly these) or `mode: free` (at least these)

**Schema Configuration Syntax:**
```yaml
plugins:
  csv_source:
    path: data.csv
    schema:
      fields: dynamic  # Accept anything - logged for audit

  # OR explicit schema:
  csv_source:
    path: data.csv
    schema:
      mode: strict      # Exactly these fields (extras rejected)
      fields:
        - id: int
        - name: str
        - score: float?  # ? = optional/nullable
```

**Trust Boundaries:**
| Plugin Type | Schema Role | On Violation |
|-------------|-------------|--------------|
| **Source** | Validates + coerces THEIR DATA | Quarantine row, continue |
| **Transform** | Contract: must output valid data | Crash (OUR CODE bug) |
| **Sink** | Expects clean data | Crash (transform bug) |

**Files to CREATE:**
- `src/elspeth/contracts/schema.py` - SchemaConfig, FieldDefinition
- `src/elspeth/plugins/schema_factory.py` - Dynamic Pydantic model creation
- Tests for above

**Files to MODIFY:**
- `src/elspeth/plugins/config_base.py` - Add DataPluginConfig with required schema
- `src/elspeth/core/landscape/schema.py` - Add schema_mode, schema_fields columns
- `src/elspeth/contracts/audit.py` - Add schema fields to Node dataclass
- `src/elspeth/core/landscape/recorder.py` - Record schema config in register_node
- All 7 plugins with hardcoded schemas (sources, sinks, transforms)

**Audit Trail:**
Schema configuration recorded at run start in `nodes` table:
- `schema_mode`: "dynamic", "strict", or "free"
- `schema_fields`: JSON array of field definitions (if explicit)

**Verification:**
- [ ] SchemaConfig and FieldDefinition types created
- [ ] Schema factory creates Pydantic models from config
- [ ] All data plugins require schema in config
- [ ] Schema choices recorded in audit trail
- [ ] Source validates + coerces at boundary
- [ ] No hardcoded `extra="allow"` schemas remain
- [ ] All tests pass

**Effort:** Medium-High (~4-6 hours)
**Dependencies:** None
**Unlocks:** WP-12 (simplified)
**Plan:** [2026-01-17-wp11.99-config-driven-schemas.md](../../2026-01-17-wp11.99-config-driven-schemas.md)

---

## WP-12: Utility Consolidation

**Goal:** Extract `get_nested_field()` utility to shared module

> **Note:** Schema consolidation is now handled by WP-11.99. This WP only extracts the `_get_nested()` utility function.

**Files:**
- `src/elspeth/plugins/utils.py` (NEW)
- `src/elspeth/plugins/transforms/field_mapper.py`

**Duplicated Code:**

`_get_nested()` exists in 4 files (3 gate files + field_mapper). Extract to utils.py:
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

**Update field_mapper.py:**
- Import `get_nested_field` from utils instead of defining locally
- (Gate files already deleted in WP-02, so only field_mapper needs updating)

**Verification:**
- [ ] `get_nested_field()` in utils.py with tests
- [ ] `_get_nested` removed from field_mapper.py
- [ ] field_mapper tests pass
- [ ] No duplicate implementations remain

**Effort:** Low (~30 minutes)
**Dependencies:** WP-11.99 (schema factory also in plugins module)
**Unlocks:** Nothing (pure cleanup)

---

## WP-13: Sink Test Rewrites

**Goal:** All sink tests use batch signature

**Files:**
- `tests/plugins/sinks/test_csv_sink.py`
- `tests/plugins/sinks/test_json_sink.py`
- `tests/plugins/sinks/test_database_sink.py`

**Note:** `test_adapters.py` is deleted in WP-04, so no adapter tests to update.

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

**MockSink in Engine Tests:**

Any engine test that needs a mock sink should create one inline or use a fixture:

```python
class MockSink:
    name = "mock"
    input_schema = DynamicSchema
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"

    def write(self, rows: list[dict], ctx) -> ArtifactDescriptor:
        self.rows_written.extend(rows)
        return ArtifactDescriptor.for_file(
            path="/tmp/mock.csv",
            content_hash="abc123",
            size_bytes=len(str(rows)),
        )

    def flush(self) -> None: pass
    def close(self) -> None: pass
    def on_start(self, ctx) -> None: pass
    def on_complete(self, ctx) -> None: pass
```

**Verification:**
- [ ] All sink plugin tests pass
- [ ] No per-row write patterns remain
- [ ] Engine tests use inline MockSink or fixture

**Effort:** Medium (~4 hours)
**Dependencies:** WP-03, WP-04
**Unlocks:** Nothing (verification)

---

## WP-14: Engine Test Rewrites

> **NOTE:** Split this WP into sub-packages when execution begins:
> - WP-14a: Fork/Coalesce tests (after WP-07, WP-08)
> - WP-14b: Gate tests (after WP-09)
> - WP-14c: Aggregation tests (after WP-06)
> - WP-14d: Integration tests (after all above)
>
> **Note:** Sink adapter tests deleted in WP-04, so no WP-14 sub-package for adapters.

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

**Track D: Gate Transition (MUST be back-to-back)**
```
WP-02 → WP-09 (no gap!)
```

**Track E: Cleanup (Anytime)**
```
WP-11 (anytime)
WP-12 (after WP-02)
```

**Final:**
```
WP-14 (after all others)
```

---

## Suggested Sprint Allocation

> **IMPORTANT:** WP-02 (delete gate plugins) and WP-09 (engine gates) MUST be executed
> back-to-back to minimize the gap where no gates exist. They are grouped in Sprint 4.

### Sprint 1: Foundation
- WP-01: Protocol & Base Class Alignment
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

### Sprint 4: Gates & Coalesce
- WP-02: Gate Plugin Deletion ← Execute first
- WP-09: Engine-Level Gates ← Execute immediately after WP-02
- WP-08: Coalesce Executor

### Sprint 5: Verification
- WP-14: Engine Test Rewrites (split into WP-14a/b/c/d/e)
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
