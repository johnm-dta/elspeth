# Telemetry Remediation Plan - North Star Document

> **Status:** Approved for Implementation
> **Created:** 2026-01-31
> **Review Board:** Architecture, Python Engineering, Quality Assurance, Systems Thinking
> **Epic:** Telemetry Subsystem Remediation

---

## Executive Summary

A comprehensive 4-perspective review of the telemetry subsystem audit revealed a **critical misdiagnosis**. The original audit identified gaps in individual plugins, but the root cause is that the **orchestrator never wires the `telemetry_emit` callback to PluginContext**.

| Finding | Original Audit | Corrected Understanding |
|---------|---------------|------------------------|
| Working plugins | 6 | **0** (all silently drop telemetry) |
| Broken plugins | 4 | **10** (all broken due to missing wiring) |
| Root cause | Individual plugin gaps | Orchestrator never wires callback |
| Fix complexity | ~200 lines across 4 plugins | **1 line** in orchestrator |

**This document serves as the authoritative reference for telemetry remediation.**

---

## Part 1: System Architecture Context

### 1.1 Telemetry vs Landscape

ELSPETH has two distinct visibility layers:

| Aspect | Telemetry | Landscape (Audit Trail) |
|--------|-----------|------------------------|
| **Purpose** | Real-time operational visibility | Legal record of what happened |
| **Persistence** | Ephemeral (streaming to external systems) | Permanent (SQLite/PostgreSQL) |
| **Completeness** | Configurable (granularity levels) | 100% complete, always |
| **Failure Mode** | Graceful degradation (log warning, continue) | Crash on anomaly |
| **Use Case** | Dashboards, alerting, debugging | Audits, lineage queries, compliance |
| **Source of Truth** | "What is happening right now" | "What happened" |

**Key Principle:** Telemetry complements but never replaces Landscape. If telemetry fails, the pipeline continues. If Landscape fails, the pipeline crashes.

### 1.2 Two-Path Emission Model

The telemetry system uses two distinct emission paths:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PATH 1: ENGINE EVENTS                                │
│                                                                             │
│  Orchestrator ──► RunStarted, RunFinished, PhaseChanged, RowCreated        │
│  Processor    ──► TransformCompleted, GateEvaluated, TokenCompleted        │
│                                                                             │
│  Purpose: DAG lifecycle visibility (when rows enter, when transforms run)   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         PATH 2: CLIENT EVENTS                                │
│                                                                             │
│  AuditedHTTPClient ──► ExternalCallCompleted (HTTP)                        │
│  AuditedLLMClient  ──► ExternalCallCompleted (LLM)                         │
│  ctx.record_call() ──► ExternalCallCompleted (SQL, Filesystem)             │
│                                                                             │
│  Purpose: External dependency health (latency, errors, rate limits)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Architectural Decision:** These paths remain separate. Unifying them would conflate concerns and make both less useful.

### 1.3 Event Types

| Event | Path | Granularity | Purpose |
|-------|------|-------------|---------|
| `RunStarted` | Engine | `lifecycle` | Pipeline begins |
| `RunFinished` | Engine | `lifecycle` | Pipeline completes |
| `PhaseChanged` | Engine | `lifecycle` | Phase transitions |
| `RowCreated` | Engine | `rows` | Row enters from source |
| `TransformCompleted` | Engine | `rows` | Transform execution finishes |
| `GateEvaluated` | Engine | `rows` | Gate makes routing decision |
| `TokenCompleted` | Engine | `rows` | Token reaches terminal state |
| `ExternalCallCompleted` | Client | `full` | HTTP/LLM/SQL call completes |

---

## Part 2: Root Cause Analysis

### 2.1 The Misdiagnosis

The original telemetry audit concluded:

> "6 plugins emit telemetry correctly via AuditedHTTPClient/AuditedLLMClient. 4 plugins have gaps."

This was **incorrect**. The audit analyzed plugins in isolation, examining whether they *use* the telemetry callback, but failed to verify whether the callback is ever *wired* in production.

### 2.2 The Actual Bug

**Location:** `src/elspeth/engine/orchestrator.py`

**Problem:** PluginContext is created without wiring `telemetry_emit`:

```python
# CURRENT CODE (BROKEN)
ctx = PluginContext(
    run_id=run_id,
    config=config.config,
    landscape=recorder,
    rate_limit_registry=self._rate_limit_registry,
    concurrency_config=self._concurrency_config,
    _batch_checkpoints=batch_checkpoints or {},
    # telemetry_emit is MISSING - defaults to no-op lambda!
)
```

**Effect:** `PluginContext.telemetry_emit` defaults to `lambda event: None`, meaning:
- ALL plugins silently drop telemetry
- Even plugins using `AuditedHTTPClient` emit nothing
- The Landscape audit trail works (separate code path)
- Only telemetry is broken

### 2.3 Why Tests Didn't Catch It

Unit tests create `PluginContext` manually with telemetry wired:

```python
# TEST CODE (WORKS)
ctx = PluginContext(
    run_id="test-run",
    telemetry_emit=mock_emit,  # ← Manually wired in tests
    # ...
)
```

Production uses `Orchestrator`, which doesn't wire it:

```python
# PRODUCTION CODE (BROKEN)
# Orchestrator.run() → creates PluginContext without telemetry_emit
```

**Pattern:** Test/production code path divergence. Tests exercise a different code path than production.

### 2.4 System Archetype: "Fixes that Fail"

```
Problem: "We need operational visibility"
         ↓
Quick Fix: Build telemetry infrastructure
         ↓
Fix Works Initially: Unit tests pass ✅
         ↓
Side Effect (DELAYED): ctx.telemetry_emit never wired in production
         ↓
Problem Returns WORSE: Think we have visibility, but we don't
         ↓
Apply More of Same: Add more exporters, more events, more docs
         ↓
Side Effect Intensifies: False confidence grows
```

---

## Part 3: Additional Findings

### 3.1 Operation Context Telemetry Bug

**Location:** `src/elspeth/plugins/context.py:335`

**Problem:** When in operation context (source/sink), telemetry uses wrong ID:

```python
# CURRENT CODE (BROKEN)
self.telemetry_emit(
    ExternalCallCompleted(
        # ...
        state_id=parent_id,  # BUG: This is operation_id, not state_id!
        # ...
    )
)
```

**Effect:** Telemetry events from source/sink operations have `state_id=operation_id`, breaking correlation with the `node_states` table.

**Options:**
1. Add `operation_id` field to `ExternalCallCompleted` event
2. Only emit telemetry for transform context (simpler)

### 3.2 Aggregation Flush Missing Telemetry

**Location:** `src/elspeth/engine/executors.py` - `AggregationExecutor.execute_flush()`

**Problem:** Aggregation flushes don't emit `TransformCompleted` telemetry, unlike regular transform execution.

**Effect:** Batch processing completions are invisible in telemetry dashboards.

### 3.3 Batch LLM Transform Issues

**OpenRouterBatchLLMTransform:**
- Uses raw `httpx.Client()` instead of `AuditedHTTPClient`
- Even after orchestrator fix, this plugin won't emit external call telemetry
- Requires refactoring to use audited client pattern

**AzureBatchLLMTransform:**
- Uses `ctx.record_call()` correctly
- Will work after orchestrator fix (assuming operation context bug is also fixed)

### 3.4 File-Based Plugins

**Finding:** CSVSource, JSONSource, CSVSink, JSONSink have no telemetry.

**Recommendation:** **Do NOT add telemetry to file-based plugins.**

Rationale:
1. File I/O is local, synchronous, and fast
2. 1M row CSV would emit 1M telemetry events (volume explosion)
3. No operational value for dashboards (not monitoring network services)
4. Landscape operations table already tracks source/sink execution

Only network-based I/O (Azure Blob, HTTP APIs, LLM calls) provides operational value.

### 3.5 False Coverage in Tests

**Location:** `tests/telemetry/test_plugin_wiring.py`

**Problem:** Tests use keyword matching, not runtime verification:

```python
# CURRENT TEST (FALSE COVERAGE)
assert "self._run_id" in source or "ctx.run_id" in source
# This passes even if run_id is captured but never used!
```

**Recommendation:** Replace with contract tests that verify actual emission.

---

## Part 4: Remediation Plan

### 4.1 Phase 1: Fix Root Cause (Critical, Immediate)

**Task 1.1: Wire telemetry_emit in Orchestrator**

```python
# src/elspeth/engine/orchestrator.py
# Find PluginContext creation and add:
ctx = PluginContext(
    run_id=run_id,
    config=config.config,
    landscape=recorder,
    rate_limit_registry=self._rate_limit_registry,
    concurrency_config=self._concurrency_config,
    _batch_checkpoints=batch_checkpoints or {},
    telemetry_emit=self._emit_telemetry,  # ← ADD THIS LINE
)
```

**Impact:** Instantly enables telemetry for all plugins using audited clients.

**Task 1.2: Add Integration Test**

```python
# tests/integration/test_telemetry_wiring.py
def test_orchestrator_wires_telemetry_to_context():
    """Verify production path wires telemetry correctly."""
    events = []

    def capture_event(event):
        events.append(event)

    # Use REAL Orchestrator, not manual PluginContext
    orch = Orchestrator(
        telemetry_manager=TelemetryManager(
            config=RuntimeTelemetryConfig(...),
            exporters=[TestExporter(on_emit=capture_event)],
        )
    )

    # Run pipeline that makes external calls
    orch.run(config_with_llm_transform)

    # Verify telemetry was emitted
    assert any(e.event_type == "external_call_completed" for e in events), \
        "Production path must emit ExternalCallCompleted"
```

### 4.2 Phase 2: Fix Operation Context Bug (High Priority)

**Task 2.1: Update ExternalCallCompleted Event**

Option A (Recommended):
```python
# telemetry/events.py
@dataclass(frozen=True, slots=True)
class ExternalCallCompleted(TelemetryEvent):
    state_id: str | None = None      # For transform context
    operation_id: str | None = None  # For source/sink context
    # ... rest of fields
```

Option B (Simpler):
```python
# context.py - Only emit telemetry for transform context
if has_state:
    self.telemetry_emit(ExternalCallCompleted(...))
# Don't emit for operation context
```

**Task 2.2: Update Emitters**

Update `context.py:record_call()` to use the correct ID field.

### 4.3 Phase 3: Fix Aggregation Flush (Medium Priority)

**Task 3.1: Add TransformCompleted Emission**

```python
# executors.py - AggregationExecutor.execute_flush()
# After successful flush:
self._emit_transform_completed(
    run_id=run_id,
    row_id=batch_row_id,
    token_id=batch_token_id,
    node_id=node_id,
    plugin_name=transform.name,
    status=CallStatus.SUCCESS,
    duration_ms=duration_ms,
    input_hash=input_hash,
    output_hash=output_hash,
)
```

### 4.4 Phase 4: Fix OpenRouterBatchLLMTransform (Medium Priority)

**Task 4.1: Refactor to Use AuditedHTTPClient**

Replace raw `httpx.Client()` with `AuditedHTTPClient`:

```python
# BEFORE (broken)
with httpx.Client(...) as client:
    response = client.post(url, json=payload)

# AFTER (correct)
http_client = AuditedHTTPClient(
    recorder=self._recorder,
    state_id=state_id,
    run_id=self._run_id,
    telemetry_emit=self._telemetry_emit,
)
response = http_client.post(url, json=payload)
```

**Note:** Must handle ThreadPoolExecutor pattern - either one client per worker or verify thread safety.

### 4.5 Phase 5: Add Contract Tests (High Priority)

**Task 5.1: Create TelemetryTestExporter**

```python
# tests/telemetry/fixtures.py
class TelemetryTestExporter(ExporterProtocol):
    """In-memory exporter for testing."""

    def __init__(self):
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def assert_event_emitted(self, event_type: str, **filters):
        matches = [e for e in self.events if e.event_type == event_type]
        assert matches, f"No {event_type} event emitted"
        return matches[0]
```

**Task 5.2: Create Plugin Contract Tests**

```python
# tests/contracts/test_telemetry_contracts.py
EXTERNAL_IO_PLUGINS = {
    "AzureLLMTransform": {"expected": ["external_call_completed"]},
    "OpenRouterLLMTransform": {"expected": ["external_call_completed"]},
    # ... all plugins with external I/O
}

@pytest.mark.parametrize("plugin_name,expected", EXTERNAL_IO_PLUGINS.items())
def test_plugin_emits_required_telemetry(plugin_name, expected, test_exporter):
    """Verify plugin ACTUALLY emits telemetry via production path."""
    # Use production Orchestrator, not manual wiring
    # Assert expected events captured
```

### 4.6 Phase 6: Verification

**Task 6.1: End-to-End Verification**

Run a full pipeline with telemetry enabled and verify:
- All expected events appear in test exporter
- Event correlation (run_id, token_id, state_id) is correct
- No silent failures or dropped events

**Task 6.2: Documentation Update**

Update `docs/guides/telemetry.md` to reflect actual working state.

**Note:** This is pre-RC-2 - we can make breaking changes freely. No gradual rollout needed.

---

## Part 5: Decision Log

### 5.1 File I/O Telemetry: REJECTED

**Decision:** Do not add telemetry to CSV/JSON sources and sinks.

**Rationale:**
- Volume explosion (1M rows = 1M events)
- No operational value (local file I/O, not network services)
- Landscape operations table provides audit coverage
- Architecture review confirmed this is appropriate

### 5.2 Two-Path Model: CONFIRMED

**Decision:** Keep engine events and client events as separate paths.

**Rationale:**
- Serve different purposes (DAG lifecycle vs external dependency health)
- Different correlation needs (token_id vs state_id)
- Unifying would conflate concerns

### 5.3 track_operation Telemetry: REJECTED

**Decision:** Do not wire telemetry into `track_operation` context manager.

**Rationale:**
- Violates separation of concerns
- Operations context is for Landscape recording
- Telemetry should come from audited clients or record_call()

---

## Part 6: Test Strategy

### 6.1 Test Pyramid

```
                    ┌─────────────────┐
                    │   E2E Tests     │  ← Full pipeline telemetry verification
                    │   (2-3 tests)   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │     Integration Tests       │  ← Orchestrator wiring, contract tests
              │        (10-15 tests)        │
              └──────────────┬──────────────┘
                             │
       ┌─────────────────────┴─────────────────────┐
       │              Unit Tests                    │  ← Individual component behavior
       │            (existing suite)                │
       └────────────────────────────────────────────┘
```

### 6.2 Required New Tests

| Test | Type | Priority | Purpose |
|------|------|----------|---------|
| `test_orchestrator_wires_telemetry_to_context` | Integration | P0 | Prevent regression of root cause |
| `test_plugin_emits_required_telemetry` (parametrized) | Contract | P1 | Verify all I/O plugins emit |
| `test_aggregation_flush_emits_transform_completed` | Integration | P2 | Verify batch telemetry |
| `test_operation_context_telemetry_correlation` | Unit | P2 | Verify correct ID in events |
| `test_end_to_end_pipeline_telemetry` | E2E | P2 | Full pipeline verification |
| `test_telemetry_queue_overflow_behavior` | Integration | P3 | Verify backpressure handling |

### 6.3 Regression Prevention

After remediation, the following tests prevent regression:

1. **Wiring test:** Fails if orchestrator stops wiring `telemetry_emit`
2. **Contract tests:** Fail if any I/O plugin stops emitting
3. **E2E test:** Fails if telemetry pipeline breaks end-to-end

---

## Part 7: Risk Assessment

### 7.1 Implementation Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Telemetry volume spike after fix | High | Medium | Deploy with `lifecycle` first |
| Breaking changes to event schema | Low | High | Add `operation_id` as optional field |
| Thread safety issues in batch refactor | Medium | High | Test with concurrent workloads |
| Exporter cost increase | High | Low | Monitor before increasing granularity |

### 7.2 Rollback Plan

If telemetry causes issues after deployment:

1. **Immediate:** Set `telemetry.enabled: false` in config
2. **Graceful:** Reduce to `granularity: lifecycle`
3. **Selective:** Disable specific exporters

---

## Part 8: Success Criteria

### 8.1 Phase 1 Complete When:

- [ ] Orchestrator wires `telemetry_emit` to PluginContext
- [ ] Integration test verifies wiring
- [ ] Existing plugins (Azure LLM, OpenRouter LLM) emit telemetry in production

### 8.2 Phase 2 Complete When:

- [ ] Operation context uses correct ID field
- [ ] Source/sink telemetry correlates correctly

### 8.3 Phase 3-4 Complete When:

- [ ] Aggregation flush emits `TransformCompleted`
- [ ] OpenRouterBatchLLMTransform uses AuditedHTTPClient

### 8.4 Phase 5-6 Complete When:

- [ ] Contract tests cover all I/O plugins
- [ ] Telemetry visible in observability platform
- [ ] No regressions in Landscape audit trail

---

## Part 9: Reference Materials

### 9.1 Key Files

| File | Purpose |
|------|---------|
| `src/elspeth/engine/orchestrator.py` | Root cause location |
| `src/elspeth/plugins/context.py` | PluginContext and record_call() |
| `src/elspeth/plugins/clients/http.py` | AuditedHTTPClient |
| `src/elspeth/plugins/clients/llm.py` | AuditedLLMClient |
| `src/elspeth/telemetry/manager.py` | TelemetryManager |
| `src/elspeth/telemetry/events.py` | Event definitions |

### 9.2 Related Documents

- `docs/architecture/telemetry-implementation-summary.md` - What's implemented
- `docs/architecture/telemetry-emission-points.md` - All emission points
- `docs/guides/telemetry.md` - User guide

### 9.3 Review Board Reports

This document incorporates findings from:

1. **Architecture Review** - Structural integrity, anti-patterns
2. **Python Engineering Review** - Code quality, thread safety, bugs
3. **Quality Assurance Review** - Test gaps, edge cases
4. **Systems Thinking Review** - Root cause, systemic patterns

---

## Appendix A: The One-Line Fix

For reference, the critical fix that enables all telemetry:

```python
# src/elspeth/engine/orchestrator.py
# In the method that creates PluginContext, add:

telemetry_emit=self._emit_telemetry,
```

This single line enables telemetry for all plugins that use audited clients or `ctx.record_call()`.

---

## Appendix B: Leverage Point Analysis

Using Meadows' 12 Leverage Points framework:

| Intervention | Level | Description | Leverage |
|--------------|-------|-------------|----------|
| Fix orchestrator wiring | 10 | System structure | **HIGHEST** |
| Add contract tests | 5 | Rules of the system | High |
| Refactor batch plugins | 6 | Information flows | Medium |
| Add documentation | 1 | Constants/parameters | Low |

**Recommendation:** Always prioritize higher-leverage interventions.

---

## Appendix C: Historical Context

### Why This Bug Existed

1. **Incremental development:** Telemetry infrastructure built before full plugin integration
2. **Test isolation:** Unit tests manually wired context, bypassing orchestrator
3. **No integration tests:** Gap between unit tests and production
4. **False confidence:** Audit report assumed plugins worked based on code patterns

### Lessons Learned

1. **Integration tests must use production code paths**
2. **Contract tests catch wiring issues that unit tests miss**
3. **Audits should verify runtime behavior, not just code patterns**
4. **Systems thinking reveals root causes that component analysis misses**
