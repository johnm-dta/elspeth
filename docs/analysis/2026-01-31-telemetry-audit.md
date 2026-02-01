# Telemetry System Audit - 2026-01-31

## Executive Summary

Comprehensive audit of telemetry usage across the codebase. The **transform-level telemetry** (audited clients) is largely conformant. However, **source and sink telemetry** is completely missing for plugins with external I/O.

**Key finding:** Everything that does external I/O should report what it's doing. No exemptions.

---

## The Correct Telemetry Pattern

```python
Plugin.on_start(ctx):
    self._run_id = ctx.run_id
    self._telemetry_emit = ctx.telemetry_emit

Plugin._get_client(state_id):
    AuditedLLMClient/AuditedHTTPClient(
        run_id=self._run_id,
        telemetry_emit=self._telemetry_emit,
    )

AuditedClient.call():
    1. Make external call
    2. recorder.record_call(...)  # Landscape FIRST
    3. try:
           self._telemetry_emit(ExternalCallCompleted(...))
       except:
           logger.warning("telemetry_emit_failed")  # Never crash
    4. Return result
```

**Key invariant:** Landscape failure = system crash. Telemetry failure = log warning, continue.

---

## Gap Summary

| Plugin | Type | External I/O | Landscape | Telemetry | Priority |
|--------|------|--------------|-----------|-----------|----------|
| AzureBatchLLMTransform | Transform | Azure Batch API | ✅ | ❌ | P1 |
| AzureBlobSource | Source | Azure Blob download | ❌ | ❌ | P1 |
| AzureBlobSink | Sink | Azure Blob upload | ❌ | ❌ | P1 |
| DatabaseSink | Sink | SQL INSERT | ❌ | ❌ | P2 |
| CSVSource | Source | Local file | N/A | N/A | - |
| CSVSink | Sink | Local file | N/A | N/A | - |
| JSONSource | Source | Local file | N/A | N/A | - |
| JSONSink | Sink | Local file | N/A | N/A | - |

**Local file I/O rationale:** These are synchronous, fast, and don't benefit from real-time dashboards. Cloud and network I/O is where visibility matters.

---

## P1: AzureBatchLLMTransform

**File:** `src/elspeth/plugins/llm/azure_batch.py`

**Current state:**
- Uses direct Azure SDK calls (not AuditedLLMClient)
- Records to Landscape via `ctx.record_call()` ✅
- No telemetry emission ❌

**External calls that need telemetry:**

| Line | Operation | CallType |
|------|-----------|----------|
| ~435 | `client.files.create()` | HTTP |
| ~475 | `client.batches.create()` | HTTP |
| ~558 | `client.batches.retrieve()` | HTTP |
| ~688 | `client.files.content()` | HTTP |
| ~890+ | Per-row LLM results | LLM |

**Fix approach:** After each `ctx.record_call()`, add telemetry emission:

```python
# After ctx.record_call() for file upload
try:
    self._telemetry_emit(
        ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id=self._run_id,
            state_id=ctx.state_id,
            call_type=CallType.HTTP,
            provider="azure",
            status=CallStatus.SUCCESS,
            latency_ms=latency_ms,
        )
    )
except Exception:
    logger.warning("telemetry_emit_failed", ...)
```

**Prerequisite:** Capture `run_id` and `telemetry_emit` in `on_start()`.

---

## P1: AzureBlobSource

**File:** `src/elspeth/plugins/azure/blob_source.py`

**Current state:**
- Uses `logger.info()` for visibility (lines 344-349)
- No Landscape recording of blob download ❌
- No telemetry emission ❌

**External call that needs both:**

| Line | Operation | CallType |
|------|-----------|----------|
| ~329 | `blob_client.download_blob()` | HTTP |

**Fix approach:**

1. Add `on_start()` to capture `run_id`, `telemetry_emit`, and `recorder`
2. After blob download, call `recorder.record_call()` for Landscape
3. After successful Landscape record, emit `ExternalCallCompleted`

---

## P1: AzureBlobSink

**File:** `src/elspeth/plugins/azure/blob_sink.py`

**Current state:**
- No Landscape recording of blob upload ❌
- No telemetry emission ❌

**External call that needs both:**

| Line | Operation | CallType |
|------|-----------|----------|
| ~445 | `blob_client.upload_blob()` | HTTP |

**Fix approach:** Same as AzureBlobSource - add `on_start()`, record + emit.

---

## P2: DatabaseSink

**File:** `src/elspeth/plugins/sinks/database_sink.py`

**Current state:**
- No Landscape recording of SQL INSERT ❌
- No telemetry emission ❌

**External call that needs both:**

| Line | Operation | CallType |
|------|-----------|----------|
| ~312 | `conn.execute(insert(...))` | SQL |

**Fix approach:** Same pattern - capture in `on_start()`, record after execute, emit telemetry.

---

## Conformant Plugins (No Changes Needed)

These 6 plugins properly use the audited client pattern:

| Plugin | File | Client |
|--------|------|--------|
| AzureLLMTransform | `plugins/llm/azure.py:224-232` | AuditedLLMClient |
| AzureMultiQueryLLMTransform | `plugins/llm/azure_multi_query.py:230-234` | AuditedLLMClient |
| OpenRouterLLMTransform | `plugins/llm/openrouter.py:202-210` | AuditedHTTPClient |
| OpenRouterMultiQueryLLMTransform | `plugins/llm/openrouter_multi_query.py:377-380` | AuditedHTTPClient |
| AzureContentSafety | `plugins/transforms/azure/content_safety.py:210-214` | AuditedHTTPClient |
| AzurePromptShield | `plugins/transforms/azure/prompt_shield.py:182-186` | AuditedHTTPClient |

All properly: capture in `on_start()`, pass to client, client emits after Landscape.

---

## Logger Usage Concerns

### Telemetry Manager Self-Logging

**File:** `src/elspeth/telemetry/manager.py`

The telemetry system logs its own failures via `logger.warning/error/critical`:

| Line | Event | Concern |
|------|-------|---------|
| 168 | Exporter failed | Logged, not telemetered |
| 189 | ALL exporters failing | Logged, not telemetered |
| 205 | Telemetry disabled | Logged, not telemetered |
| 283 | Events dropped (backpressure) | Logged, not telemetered |

**Issue:** When telemetry fails, you're logging to the regular log. If you're relying on telemetry for operational visibility, you have a blind spot for telemetry system health.

**Possible fix:** Last-resort exporter (console/file) that never fails for system health events.

### Correct Logger Usage (No Action Needed)

- `plugins/clients/http.py:338, 384` - "telemetry_emit_failed" after successful Landscape
- `plugins/clients/llm.py:359, 417` - Same pattern
- `plugins/context.py:313-316` - Tier 3 boundary warnings (non-canonical serialization)
- `engine/executors.py:1338` - Checkpoint size warnings

---

## Implementation Checklist

### Phase 1: Source/Sink Telemetry Infrastructure

- [ ] Add `TelemetryEmitCallback` type to base source/sink classes
- [ ] Update `on_start()` signature to accept recorder reference (or use ctx)
- [ ] Define pattern for sources (different from transforms - no state_id)

### Phase 2: Fix P1 Gaps

- [ ] **AzureBatchLLMTransform**: Add telemetry emission after `ctx.record_call()`
- [ ] **AzureBlobSource**: Add Landscape recording + telemetry for blob download
- [ ] **AzureBlobSink**: Add Landscape recording + telemetry for blob upload

### Phase 3: Fix P2 Gaps

- [ ] **DatabaseSink**: Add Landscape recording + telemetry for SQL INSERT

### Phase 4: Test Coverage

- [ ] Update `tests/telemetry/test_plugin_wiring.py` to cover sources/sinks
- [ ] Add integration tests verifying telemetry emission

---

## Design Question: Source/Sink State IDs

Transforms have `state_id` from the node_states table. Sources and sinks don't have per-row state tracking in the same way.

Options:
1. Use `run_id` only for source/sink telemetry (no state_id)
2. Create synthetic state_id for source/sink operations
3. Add source_id/sink_id columns to the calls table

Recommendation: Option 1 for now - sources/sinks operate at run level, not row level. The `state_id` field in `ExternalCallCompleted` can be optional/empty for these cases.

---

## Files Reference

### Telemetry Core

- `src/elspeth/telemetry/manager.py` - Central hub, backpressure, failure isolation
- `src/elspeth/telemetry/events.py` - ExternalCallCompleted, lifecycle events
- `src/elspeth/telemetry/filtering.py` - LIFECYCLE/ROWS/FULL granularity
- `src/elspeth/telemetry/exporters/` - Console, OTLP, Azure Monitor, Datadog

### Audited Clients (Reference Implementation)

- `src/elspeth/plugins/clients/base.py` - TelemetryEmitCallback type
- `src/elspeth/plugins/clients/llm.py:338-366` - LLM emission pattern
- `src/elspeth/plugins/clients/http.py:318-345` - HTTP emission pattern

### Gap Files

- `src/elspeth/plugins/llm/azure_batch.py` - P1: Add telemetry emission
- `src/elspeth/plugins/azure/blob_source.py` - P1: Add both Landscape + telemetry
- `src/elspeth/plugins/azure/blob_sink.py` - P1: Add both Landscape + telemetry
- `src/elspeth/plugins/sinks/database_sink.py` - P2: Add both Landscape + telemetry

### Context Flow

- `src/elspeth/plugins/context.py:119` - `telemetry_emit` field (default no-op)
- `src/elspeth/engine/orchestrator.py` - Creates PluginContext
