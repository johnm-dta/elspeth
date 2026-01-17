# Plugin Protocol Gap Analysis

> **Date:** 2026-01-17
> **Contract Version:** plugin-protocol.md v1.1
> **Status:** DRAFT - Comprehensive gap analysis

## Executive Summary

This document analyzes the current ELSPETH implementation against the Plugin Protocol Contract v1.1. The analysis was conducted by systematically reviewing every source file against contract requirements.

**Overall Assessment:** The core infrastructure is solid, but significant gaps exist between the contract specification and implementation. Most gaps fall into two categories:

1. **Protocol debt** - Contract finalized after initial implementation
2. **Phase 4 features** - DAG execution (fork/join) partially implemented

| Category | Compliance | Blocking Issues |
|----------|------------|-----------------|
| Plugin Lifecycles | 100% | None |
| Audit Infrastructure | 85% | Idempotency keys, lifecycle events |
| Result Contracts | 95% | Sink.write() signature |
| Terminal States | 85% | QUARANTINED, COALESCED unreachable |
| Gates | 0% | Wrong architecture (plugin vs engine) |
| Aggregation | 50% | Missing triggers, output modes |
| Fork | 50% | Tokens created, not executed |
| Coalesce | 0% | Not implemented |
| Sink Plugins | 30% | Missing ArtifactDescriptor |
| Transform Plugins | 60% | Missing metadata |

---

## Critical Gaps (P0)

### GAP-001: Gates Implemented as Plugins Instead of Engine Operations

**Contract Reference:** Lines 530-603

**What the contract requires:**
- Gates are **system operations**, not user plugins
- Config-driven with expression language
- Safe expression parser (NOT Python eval)
- Support for `continue`, `route_to_sink`, `fork_to_paths`

**Current implementation:**
- Three plugin classes: `FilterGate`, `FieldMatchGate`, `ThresholdGate`
- Hardcoded comparison logic per plugin
- No expression parser
- Cannot compose conditions

**Gap details:**

| Requirement | Contract | Implementation |
|-------------|----------|----------------|
| Architecture | Engine-level operation | Plugin class |
| Configuration | YAML conditions | Python code |
| Expression safety | Restricted AST parser | N/A |
| Composability | `row['a'] > 0.8 and row['b'] == 'x'` | Single comparison only |
| Fork support | `fork_to: [path1, path2]` | Not accessible |

**Files affected:**
- `src/elspeth/plugins/gates/filter_gate.py`
- `src/elspeth/plugins/gates/field_match_gate.py`
- `src/elspeth/plugins/gates/threshold_gate.py`

**Recommended action:** Create engine-level gate infrastructure with safe expression parser. Deprecate plugin gates.

---

### GAP-002: Sink.write() Signature Mismatch

**Contract Reference:** Lines 373-386

**What the contract requires:**
```python
def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
    """Receive rows and return proof of work.

    MUST return ArtifactDescriptor with content_hash and size_bytes (REQUIRED for audit)
    """
```

**Current implementation (all sinks):**
```python
def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
    """Write a single row."""
```

**Gap details:**

| Requirement | Contract | Implementation |
|-------------|----------|----------------|
| Input | `rows: list[dict]` (batch) | `row: dict` (single) |
| Return | `ArtifactDescriptor` | `None` |
| content_hash | Required for audit | Missing |
| size_bytes | Required for audit | Missing |

**Files affected:**
- `src/elspeth/plugins/sinks/csv_sink.py`
- `src/elspeth/plugins/sinks/json_sink.py`
- `src/elspeth/plugins/sinks/database_sink.py`
- `src/elspeth/plugins/protocols.py` (SinkProtocol)
- `src/elspeth/plugins/base.py` (BaseSink)

**Audit impact:** Cannot prove what was written. Audit trail incomplete.

---

### GAP-003: Coalesce Not Implemented

**Contract Reference:** Lines 677-753

**What the contract requires:**
- Merge tokens from parallel fork paths
- Policies: `require_all`, `quorum`, `best_effort`, `first`
- Merge strategies: `union`, `nested`, `select`
- Timeout handling
- Child tokens marked `COALESCED`

**Current implementation:**
- `CoalesceProtocol` defined in `protocols.py`
- `CoalescePolicy` enum defined
- `LandscapeRecorder.coalesce_tokens()` exists
- **No `CoalesceExecutor` class**
- **No execution path in `RowProcessor`**
- **COALESCED terminal state unreachable**

**Gap details:**

| Component | Status |
|-----------|--------|
| Protocol definition | Complete |
| Policy enum | Complete |
| Recorder method | Complete |
| Executor | Missing |
| RowProcessor path | Missing |
| Policy enforcement | Missing |
| Merge strategy impl | Missing |

**Files to create:**
- `src/elspeth/engine/coalesce_executor.py`

**Files to modify:**
- `src/elspeth/engine/processor.py` - Add coalesce handling

---

### GAP-004: Fork Execution Incomplete

**Contract Reference:** Lines 604-676

**What the contract requires:**
- Fork creates N child tokens with same row_id
- Each child assigned to specific path
- Parent marked `FORKED`
- **Children execute through their paths**

**Current implementation:**
- `TokenManager.fork_token()` creates children correctly
- Children recorded in audit trail
- Parent marked `FORKED`
- **Children never executed** (orphaned)

**Evidence (processor.py line 91):**
```python
# NOTE: This implementation handles LINEAR pipelines only. For DAG support
# (fork/join), this needs a work queue that processes child tokens from forks.
```

**Gap:** No work queue to process fork children. DAG execution blocked.

---

### GAP-005: Missing Plugin Metadata

**Contract Reference:** Lines 155-162, 251-259, 358-366

**What the contract requires:**
All plugins must declare:
- `determinism: Determinism`
- `plugin_version: str`
- `on_start(ctx)` lifecycle hook
- `on_complete(ctx)` lifecycle hook

**Current implementation:**

| Plugin | determinism | plugin_version | on_start | on_complete |
|--------|-------------|----------------|----------|-------------|
| CSVSink | Missing | Missing | Missing | Missing |
| JSONSink | Missing | Missing | Missing | Missing |
| DatabaseSink | Missing | Missing | Missing | Missing |
| PassThrough | Missing | Missing | Missing | Missing |
| FieldMapper | Missing | Missing | Missing | Missing |

**Contract quote (line 141):**
> "All lifecycle hooks are REQUIRED in the protocol, even if implementation is `pass`."

**Files affected:**
- All files in `src/elspeth/plugins/sinks/`
- All files in `src/elspeth/plugins/transforms/`

---

### GAP-006: SourceProtocol Missing Required Attributes

**Contract Reference:** Lines 155-162

**What the contract requires:**
```python
name: str
output_schema: type[PluginSchema]
node_id: str | None
determinism: Determinism           # REQUIRED
plugin_version: str                # REQUIRED
```

**Current implementation (protocols.py lines 52-54):**
```python
name: str
output_schema: type["PluginSchema"]
node_id: str | None
# determinism: MISSING
# plugin_version: MISSING
```

**Note:** SourceProtocol is the ONLY protocol missing these attributes. All others (Transform, Gate, Aggregation, Sink) include them.

**Files affected:**
- `src/elspeth/plugins/protocols.py`
- `src/elspeth/plugins/base.py`

---

## High Priority Gaps (P1)

### GAP-007: Aggregation Trigger Logic Missing

**Contract Reference:** Lines 768-784

**What the contract requires:**
```yaml
trigger:
  count: 1000           # Fire after 1000 rows
  timeout: 1h           # Or after 1 hour
  condition: "row['type'] == 'flush_signal'"
  # end_of_source: implicit
```

**Current implementation:**
- Batch infrastructure works (create, add member, transition states)
- Hardcoded `trigger_reason="threshold"`
- No trigger configuration schema
- No `AggregationSettings` in config.py

| Trigger Type | Status |
|--------------|--------|
| count | Not implemented |
| timeout | Not implemented |
| condition | Not implemented |
| end_of_source | Implicit only |

**Files affected:**
- `src/elspeth/core/config.py` - Missing AggregationSettings
- `src/elspeth/engine/orchestrator.py` - No trigger evaluation

---

### GAP-008: Aggregation Output Modes Missing

**Contract Reference:** Lines 786-793

**What the contract requires:**

| Mode | Behavior |
|------|----------|
| `single` | Emit one aggregated result |
| `passthrough` | Release all accumulated tokens |
| `transform` | Apply transform to batch |

**Current implementation:** Only implicit `single` mode.

---

### GAP-009: Missing Audit Capabilities

**Contract Reference:** Various sections

| Audit Requirement | Status | Contract Lines |
|-------------------|--------|----------------|
| Plugin lifecycle events | Missing | 196-212, 287-295 |
| Idempotency key tracking | Missing | 483-489 |
| External call recording | Partial (read-only) | N/A |
| Coalesce policy/strategy | Missing | 699-744 |
| Batch trigger details | Missing | 830-836 |
| Source completion metrics | Missing | 236-240 |

**Idempotency key format (contract):** `{run_id}:{token_id}:{sink_name}`

**Files affected:**
- `src/elspeth/core/landscape/recorder.py`
- `src/elspeth/core/landscape/schema.py`

---

### GAP-010: Terminal States Unreachable

**Contract Reference:** Lines 800, 827-828, implicit

| State | Status | Issue |
|-------|--------|-------|
| COMPLETED | Works | - |
| ROUTED | Works | - |
| FORKED | Works | - |
| CONSUMED_IN_BATCH | Works | - |
| FAILED | Works | - |
| COALESCED | Unreachable | No execution path |
| QUARANTINED | Unreachable | No quarantine logic |

**Files affected:**
- `src/elspeth/engine/processor.py`

---

## Medium Priority Gaps (P2)

### GAP-011: Determinism Enum Extended Beyond Spec

**Contract (3 levels):**
- `DETERMINISTIC`
- `NON_DETERMINISTIC`
- `EXTERNAL`

**Implementation (6 levels):**
- `DETERMINISTIC`
- `SEEDED`
- `IO_READ`
- `IO_WRITE`
- `EXTERNAL_CALL`
- `NON_DETERMINISTIC`

**Assessment:** Enhancement, not breaking. Documentation should be updated.

---

### GAP-012: Retry Metadata Incomplete

**Current:**
- Attempt counter exists
- Each attempt is separate node state

**Missing:**
- Backoff parameters (delay_ms, base_delay, jitter)
- Link between retry attempts ("state_id_2 is retry of state_id_1")

---

### GAP-013: Error Categorization Missing

**Current:**
- Errors stored as `error_json` (unstructured)

**Missing:**
- Error type/category field
- Explicit retryable flag in NodeState
- Error code classification

---

## What's Working Correctly

### Fully Compliant Areas

1. **Plugin Lifecycles**
   - Source: `__init__ → on_start → load → on_complete → close`
   - Transform: `__init__ → on_start → process(×N) → on_complete → close`
   - Sink: `__init__ → on_start → write(×N) → flush → on_complete → close`
   - All correctly implemented in orchestrator

2. **Core Result Contracts**
   - `TransformResult` - Perfect match with factory methods
   - `GateResult` - Correctly implemented
   - `RoutingAction` - All factory methods work
   - `ArtifactDescriptor` - Correctly defined (sinks don't use it)

3. **Token Identity**
   - `row_id` vs `token_id` distinction maintained
   - Fork lineage tracked with `parent_token_id`
   - `fork_group_id` and `join_group_id` recorded

4. **Audit Database Schema**
   - All tables properly defined
   - State transitions tracked
   - Indexes for efficient queries

5. **Gate Routing (Executor Level)**
   - Route resolution works
   - Routing events recorded
   - CONTINUE/ROUTE/FORK_TO_PATHS handled

6. **Batch Infrastructure**
   - Batch creation and membership
   - DRAFT → EXECUTING → COMPLETED lifecycle
   - batch_members table properly designed

---

## Implementation Priority

### Phase 1: Protocol Compliance (Low Effort, High Impact)

| Task | Effort | Files |
|------|--------|-------|
| Add determinism to SourceProtocol | Low | protocols.py |
| Add plugin_version to SourceProtocol | Low | protocols.py |
| Add lifecycle hooks to all plugins | Low | sinks/*.py, transforms/*.py |
| Add determinism to all plugins | Low | sinks/*.py, transforms/*.py |

### Phase 2: Sink Contract Fix (Medium Effort, Critical)

| Task | Effort | Files |
|------|--------|-------|
| Change write() signature to batch | Medium | base.py, protocols.py |
| Return ArtifactDescriptor | Medium | All sinks |
| Implement content hashing | Medium | All sinks |

### Phase 3: Aggregation Completion (Medium Effort)

| Task | Effort | Files |
|------|--------|-------|
| Create AggregationSettings | Medium | config.py |
| Implement trigger types | Medium | orchestrator.py |
| Implement output modes | Medium | aggregation executor |

### Phase 4: DAG Execution (High Effort)

| Task | Effort | Files |
|------|--------|-------|
| Engine-level gate with parser | High | New file |
| Fork work queue | High | processor.py |
| CoalesceExecutor | High | New file |
| QUARANTINED execution path | Medium | processor.py |

---

## Files Requiring Changes

### Must Modify

| File | Changes |
|------|---------|
| `plugins/protocols.py` | Add SourceProtocol attributes |
| `plugins/base.py` | Add BaseSource attributes, fix BaseSink.write() |
| `plugins/sinks/csv_sink.py` | Full rewrite for contract |
| `plugins/sinks/json_sink.py` | Full rewrite for contract |
| `plugins/sinks/database_sink.py` | Full rewrite for contract |
| `plugins/transforms/passthrough.py` | Add metadata + hooks |
| `plugins/transforms/field_mapper.py` | Add metadata + hooks |
| `engine/processor.py` | Work queue for DAG |
| `core/config.py` | AggregationSettings |

### Must Create

| File | Purpose |
|------|---------|
| `engine/expression_parser.py` | Safe gate expression evaluation |
| `engine/coalesce_executor.py` | Coalesce logic |

### Should Delete (Eventually)

| File | Reason |
|------|--------|
| `plugins/gates/filter_gate.py` | Replace with engine gates |
| `plugins/gates/field_match_gate.py` | Replace with engine gates |
| `plugins/gates/threshold_gate.py` | Replace with engine gates |

---

## Risk Assessment

| Gap | Risk if Unfixed | Mitigation |
|-----|-----------------|------------|
| Sink signature | Audit incomplete, cannot prove writes | High priority fix |
| Gates as plugins | Cannot compose conditions | Defer to Phase 4 |
| Coalesce missing | Cannot execute join pipelines | Defer to Phase 4 |
| Fork incomplete | Cannot execute DAG pipelines | Defer to Phase 4 |
| Missing metadata | Protocol violation at registration | Quick fix |

---

## Appendix: Contract Sections Reviewed

| Section | Lines | Status |
|---------|-------|--------|
| Overview | 1-46 | Compliant |
| Core Principles | 48-144 | Compliant |
| Source | 149-242 | GAP-006 |
| Transform | 244-348 | Compliant |
| Sink | 350-496 | GAP-002 |
| Gate | 530-603 | GAP-001 |
| Fork | 604-676 | GAP-004 |
| Coalesce | 677-753 | GAP-003 |
| Aggregation | 754-836 | GAP-007, GAP-008 |
| Exception Handling | 856-867 | Compliant |
| Determinism | 870-884 | GAP-011 |
| Engine Concerns | 888-926 | Compliant |
