# Telemetry Emission Points - Complete Audit

> **Generated:** 2026-01-31
> **Scope:** All telemetry emission points across the ELSPETH codebase
> **Purpose:** Document where, when, and what telemetry events are emitted

---

## Executive Summary

| Component | Emission Points | Status |
|-----------|-----------------|--------|
| Engine (Orchestrator) | 8 | ✅ Complete |
| Engine (Processor) | 11 | ✅ Complete |
| Source Plugins | 1 of 4 | ⚠️ Gaps |
| Transform Plugins | 2 of 9 | ✅ Appropriate (others don't need) |
| Sink Plugins | 1 of 3 | ⚠️ Gaps |
| LLM Plugins | 4 of 6 | ❌ Critical gaps |
| Audited Clients | 4 | ✅ Complete |
| Core/CLI/TUI | 0 | ✅ Appropriate |

**Total Active Emission Points:** ~30 (engine + clients)
**Critical Gaps:** 4 plugins missing telemetry for external I/O

---

## Part 1: Event Types Available

### Defined in `telemetry/events.py`

| Event | Purpose | Granularity Level |
|-------|---------|-------------------|
| `RunStarted` | Pipeline run begins | `lifecycle` |
| `RunFinished` | Pipeline completes (success/failure) | `lifecycle` |
| `PhaseChanged` | Phase transitions (CONFIG→GRAPH→SOURCE→PROCESS→EXPORT) | `lifecycle` |
| `RowCreated` | Row enters pipeline from source | `rows` |
| `ExternalCallCompleted` | External call completes (HTTP, LLM, SQL) | `full` |

### Defined in `contracts/events.py`

| Event | Purpose | Granularity Level |
|-------|---------|-------------------|
| `TransformCompleted` | Transform execution finishes | `rows` |
| `GateEvaluated` | Gate makes routing decision | `rows` |
| `TokenCompleted` | Token reaches terminal state | `rows` |

---

## Part 2: Engine Emission Points

### Orchestrator (`engine/orchestrator.py`)

**8 emission points** - All lifecycle and row-level events

| Line | Event | Trigger Condition |
|------|-------|-------------------|
| 574 | `RunStarted` | After `recorder.begin_run()` succeeds |
| 609 | `RunFinished` | After successful `recorder.finalize_run(COMPLETED)` |
| 638 | `PhaseChanged` | EXPORT phase entry (setting export status) |
| 719 | `RunFinished` | After exception caught, `recorder.finalize_run(FAILED)` |
| 824 | `PhaseChanged` | GRAPH phase entry (before node registration) |
| 1081 | `PhaseChanged` | SOURCE phase entry (before source initialization) |
| 1127 | `PhaseChanged` | PROCESS phase entry (before row processing) |
| 1171 | `RowCreated` | After creating token for each source row |

**Pattern:**
```python
def _emit_telemetry(self, event: TelemetryEvent) -> None:
    if self._telemetry is not None:
        self._telemetry.handle_event(event)
```

---

### Processor (`engine/processor.py`)

**11 emission points** - All row-level transform/gate/outcome events

| Line | Event | Trigger Condition |
|------|-------|-------------------|
| 522 | `TokenCompleted` | Token routed to sink, outcome=FAILED |
| 824 | `TokenCompleted` | Gate routing failure, outcome=FAILED |
| 854 | `TokenCompleted` | Batch aggregation accepted, outcome=CONSUMED_IN_BATCH |
| 1041 | `TokenCompleted` | Batch trigger fired, outcome=CONSUMED_IN_BATCH |
| 1122 | `TokenCompleted` | Batch flush at end-of-source, outcome=CONSUMED_IN_BATCH |
| 1523 | `TokenCompleted` | Aggregation error, outcome=FAILED |
| 1599 | `GateEvaluated` | After config gate execution |
| 1697 | `TransformCompleted` | After transform execution with retry |
| 1712 | `TokenCompleted` | Transform execution failure, outcome=FAILED |
| 1736 | `TokenCompleted` | Transform produces validation error, outcome=QUARANTINED |
| 1848 | `GateEvaluated` | After config gate execution (alternate path) |

**Helper Methods:**
- `_emit_transform_completed()` - Lines 185-223
- `_emit_gate_evaluated()` - Lines 225-262
- `_emit_token_completed()` - Lines 264-295

---

### Other Engine Components (No Telemetry)

| File | Reason |
|------|--------|
| `coalesce_executor.py` | Records to Landscape directly; no separate telemetry |
| `artifacts.py` | Re-export only |
| `retry.py` | Uses tenacity; retry tracking via Landscape |
| `tokens.py` | Delegates to LandscapeRecorder |
| `triggers.py` | Pure evaluation logic; no external I/O |

---

## Part 3: Plugin Emission Points

### Source Plugins

| Plugin | File | External I/O | Telemetry | Status |
|--------|------|--------------|-----------|--------|
| **CSVSource** | `sources/csv_source.py` | File read | ❌ None | **GAP** |
| **JSONSource** | `sources/json_source.py` | File read | ❌ None | **GAP** |
| **NullSource** | `sources/null_source.py` | None | N/A | ✅ OK |
| **AzureBlobSource** | `azure/blob_source.py` | Azure Blob download | ✅ Via `record_call()` | ✅ OK |

**Gap Details:**
- CSVSource/JSONSource perform file I/O without `record_call()` or telemetry emission
- Local file operations have no operational visibility
- AzureBlobSource correctly uses `ctx.record_call()` which triggers `ExternalCallCompleted`

---

### Transform Plugins

| Plugin | File | External I/O | Telemetry | Status |
|--------|------|--------------|-----------|--------|
| **PassThrough** | `transforms/passthrough.py` | None | N/A | ✅ OK |
| **BatchReplicate** | `transforms/batch_replicate.py` | None | N/A | ✅ OK |
| **BatchStats** | `transforms/batch_stats.py` | None | N/A | ✅ OK |
| **KeywordFilter** | `transforms/keyword_filter.py` | None | N/A | ✅ OK |
| **Truncate** | `transforms/truncate.py` | None | N/A | ✅ OK |
| **JSONExplode** | `transforms/json_explode.py` | None | N/A | ✅ OK |
| **FieldMapper** | `transforms/field_mapper.py` | None | N/A | ✅ OK |
| **AzureContentSafety** | `azure/content_safety.py` | Azure API | ✅ Via `AuditedHTTPClient` | ✅ OK |
| **AzurePromptShield** | `azure/prompt_shield.py` | Azure API | ✅ Via `AuditedHTTPClient` | ✅ OK |

**Key Insight:** Transforms without external I/O correctly have no telemetry. Azure transforms use `AuditedHTTPClient` which emits `ExternalCallCompleted` automatically.

---

### Sink Plugins

| Plugin | File | External I/O | Telemetry | Status |
|--------|------|--------------|-----------|--------|
| **CSVSink** | `sinks/csv_sink.py` | File write | ❌ None | **GAP** |
| **JSONSink** | `sinks/json_sink.py` | File write | ❌ None | **GAP** |
| **DatabaseSink** | `sinks/database_sink.py` | SQL INSERT | ✅ Via `record_call()` | ✅ OK |

**Gap Details:**
- CSVSink/JSONSink perform file writes with no telemetry
- DatabaseSink correctly uses `ctx.record_call(CallType.SQL)` which triggers `ExternalCallCompleted`

---

### LLM Plugins

| Plugin | File | External I/O | Telemetry | Status |
|--------|------|--------------|-----------|--------|
| **AzureLLMTransform** | `llm/azure.py` | Azure OpenAI | ✅ Via `AuditedLLMClient` | ✅ OK |
| **AzureMultiQueryLLMTransform** | `llm/azure_multi_query.py` | Azure OpenAI | ✅ Via `AuditedLLMClient` | ✅ OK |
| **AzureBatchLLMTransform** | `llm/azure_batch.py` | Azure Batch API | ❌ Has `record_call()` but no telemetry emission | **CRITICAL GAP** |
| **OpenRouterLLMTransform** | `llm/openrouter.py` | OpenRouter API | ✅ Via `AuditedHTTPClient` | ✅ OK |
| **OpenRouterMultiQueryLLMTransform** | `llm/openrouter_multi_query.py` | OpenRouter API | ✅ Via `AuditedHTTPClient` | ✅ OK |
| **OpenRouterBatchLLMTransform** | `llm/openrouter_batch.py` | OpenRouter API | ❌ No `record_call()`, no telemetry | **CRITICAL GAP** |

**Critical Gap Details:**
- **AzureBatchLLMTransform**: Uses `ctx.record_call()` but `record_call()` in operations context doesn't emit telemetry (different code path than transform context)
- **OpenRouterBatchLLMTransform**: Uses raw `httpx.Client()` with no instrumentation at all - complete blind spot

---

## Part 4: Client Emission Points

### AuditedHTTPClient (`plugins/clients/http.py`)

**2 emission points** - All HTTP calls

| Lines | Event | When |
|-------|-------|------|
| 322-335 | `ExternalCallCompleted` | After HTTP request succeeds |
| 368-381 | `ExternalCallCompleted` | After HTTP request fails |

**Fields Emitted:**
```python
ExternalCallCompleted(
    timestamp=datetime.now(UTC),
    run_id=self._run_id,
    state_id=self._state_id,
    call_type=CallType.HTTP,
    provider=self._extract_provider(full_url),  # e.g., "api.openai.com"
    status=CallStatus.SUCCESS | ERROR,
    latency_ms=latency_ms,
    request_hash=stable_hash(request_data),
    response_hash=stable_hash(response_data),
    token_usage=None,  # HTTP calls don't have tokens
)
```

---

### AuditedLLMClient (`plugins/clients/llm.py`)

**2 emission points** - All LLM calls

| Lines | Event | When |
|-------|-------|------|
| 342-355 | `ExternalCallCompleted` | After LLM chat completion succeeds |
| 401-414 | `ExternalCallCompleted` | After LLM chat completion fails |

**Fields Emitted:**
```python
ExternalCallCompleted(
    timestamp=datetime.now(UTC),
    run_id=self._run_id,
    state_id=self._state_id,
    call_type=CallType.LLM,
    provider=self._provider,  # "azure" or "openai"
    status=CallStatus.SUCCESS | ERROR,
    latency_ms=latency_ms,
    request_hash=stable_hash(request_data),
    response_hash=stable_hash(response_data),
    token_usage=usage,  # {prompt_tokens, completion_tokens}
)
```

---

## Part 5: PluginContext Integration

### telemetry_emit Callback (`plugins/context.py:122`)

```python
telemetry_emit: Callable[[Any], None] = field(default=lambda event: None)
```

**Design:**
- Always present (never None) - defaults to no-op lambda
- Orchestrator provides real callback or no-op
- Plugins call without None checks
- Safe to call even when telemetry is disabled

---

### record_call() Method (`plugins/context.py:225-357`)

**Dual Recording:**
1. Records to Landscape audit trail (legal record)
2. Emits `ExternalCallCompleted` telemetry (operational visibility)

**XOR Enforcement:**
- Exactly ONE of `state_id` or `operation_id` must be set
- `state_id`: Transform calls (node_states table)
- `operation_id`: Source/sink calls (operations table)

**Telemetry Emission (lines 331-355):**
```python
try:
    self.telemetry_emit(
        ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id=self.run_id,
            state_id=parent_id,
            call_type=call_type,
            provider=provider,
            status=status,
            latency_ms=latency_ms or 0.0,
            request_hash=stable_hash(request_data),
            response_hash=stable_hash(response_data) if response_data else None,
            token_usage=token_usage,
        )
    )
except Exception as tel_err:
    logger.warning("telemetry_emit_failed", error=str(tel_err), ...)
```

---

## Part 6: Components Without Telemetry

### Core Subsystems

| Subsystem | Reason |
|-----------|--------|
| `core/landscape/` | IS the audit trail; telemetry is separate concern |
| `core/checkpoint/` | Internal persistence; no external I/O |
| `core/rate_limit/` | Internal coordination; no external I/O |
| `core/retention/` | Background cleanup; no real-time visibility needed |
| `core/security/` | HMAC fingerprinting; no external I/O |
| `core/canonical.py` | Pure functions; no I/O |
| `core/dag.py` | Graph construction; no I/O |
| `core/events.py` | Event bus; internal coordination |
| `core/payload_store.py` | Internal storage; covered by Landscape |

### CLI and TUI

| Component | Reason |
|-----------|--------|
| `cli.py` | User interface; not pipeline execution |
| `tui/` | User interface; not pipeline execution |

---

## Part 7: Gap Summary

### Critical Gaps (Missing External I/O Telemetry)

| Plugin | I/O Type | Impact | Fix Approach |
|--------|----------|--------|--------------|
| **OpenRouterBatchLLMTransform** | HTTP | Complete blind spot | Add `AuditedHTTPClient` usage |
| **AzureBatchLLMTransform** | HTTP | Audit recorded, no telemetry | Wire telemetry into operation context |

### Moderate Gaps (Local File I/O)

| Plugin | I/O Type | Impact | Fix Approach |
|--------|----------|--------|--------------|
| **CSVSource** | File read | No file load visibility | Consider `record_call(FILESYSTEM)` |
| **JSONSource** | File read | No file load visibility | Consider `record_call(FILESYSTEM)` |
| **CSVSink** | File write | No file write visibility | Consider `record_call(FILESYSTEM)` |
| **JSONSink** | File write | No file write visibility | Consider `record_call(FILESYSTEM)` |

### Known Engine Gaps

| Component | Issue | Priority |
|-----------|-------|----------|
| Aggregation flush | `execute_flush()` doesn't emit `TransformCompleted` | P3 |
| Coalesce operations | No telemetry for merge operations | P4 |
| Operation lifecycle | `track_operation` context has no telemetry | P3 |

---

## Part 8: Telemetry Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE THREAD                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ORCHESTRATOR                                                               │
│  ├─ RunStarted ──────────────────────────────────────────┐                 │
│  ├─ PhaseChanged (CONFIG) ───────────────────────────────┤                 │
│  ├─ PhaseChanged (GRAPH) ────────────────────────────────┤                 │
│  ├─ PhaseChanged (SOURCE) ───────────────────────────────┤                 │
│  ├─ RowCreated (per row) ────────────────────────────────┤                 │
│  ├─ PhaseChanged (PROCESS) ──────────────────────────────┤                 │
│  ├─ PhaseChanged (EXPORT) ───────────────────────────────┤                 │
│  └─ RunFinished ─────────────────────────────────────────┤                 │
│                                                          │                  │
│  PROCESSOR                                               │                  │
│  ├─ TransformCompleted ──────────────────────────────────┤                 │
│  ├─ GateEvaluated ───────────────────────────────────────┤                 │
│  └─ TokenCompleted ──────────────────────────────────────┤                 │
│                                                          │                  │
│  AUDITED CLIENTS (via plugins)                           │                  │
│  ├─ ExternalCallCompleted (HTTP) ────────────────────────┤                 │
│  └─ ExternalCallCompleted (LLM) ─────────────────────────┤                 │
│                                                          ▼                  │
│                                              ┌───────────────────────┐      │
│                                              │  TelemetryManager     │      │
│                                              │  .handle_event()      │      │
│                                              │  (thread-safe queue)  │      │
│                                              └───────────┬───────────┘      │
└──────────────────────────────────────────────────────────┼──────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXPORT THREAD                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Queue ──► Granularity Filter ──► Exporter Dispatch                        │
│                                         │                                   │
│                    ┌────────────────────┼────────────────────┐             │
│                    ▼                    ▼                    ▼              │
│             ┌──────────┐         ┌──────────┐         ┌──────────┐         │
│             │ Console  │         │   OTLP   │         │ Datadog  │         │
│             │ Exporter │         │ Exporter │         │ Exporter │         │
│             └──────────┘         └──────────┘         └──────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 9: Quick Reference

### Find All Emission Points

```bash
# All telemetry_emit calls
rg "telemetry_emit\(" src/elspeth/

# All TelemetryEvent instantiation
rg "(RunStarted|RunFinished|PhaseChanged|RowCreated|ExternalCallCompleted|TransformCompleted|GateEvaluated|TokenCompleted)\(" src/elspeth/

# All handle_event calls
rg "handle_event\(" src/elspeth/
```

### Key Files

| File | Purpose |
|------|---------|
| `engine/orchestrator.py` | Lifecycle events (RunStarted, RunFinished, PhaseChanged, RowCreated) |
| `engine/processor.py` | Row-level events (TransformCompleted, GateEvaluated, TokenCompleted) |
| `plugins/clients/http.py` | HTTP call telemetry (ExternalCallCompleted) |
| `plugins/clients/llm.py` | LLM call telemetry (ExternalCallCompleted) |
| `plugins/context.py` | record_call() unified recording + telemetry |
| `telemetry/events.py` | Event definitions |
| `telemetry/manager.py` | Queue management and dispatch |
