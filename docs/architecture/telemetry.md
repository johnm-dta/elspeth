# Telemetry Architecture

> **Status:** RC-3
> **Last Updated:** 2026-02-14
> **Purpose:** Operational visibility into pipeline execution - real-time monitoring, not audit trail

---

## Overview

### Telemetry vs Landscape

| Aspect | Telemetry | Landscape (Audit Trail) |
|--------|-----------|------------------------|
| **Purpose** | Real-time operational visibility | Legal record of what happened |
| **Persistence** | Ephemeral (streaming to external systems) | Permanent (SQLite/PostgreSQL) |
| **Completeness** | Configurable (granularity levels) | 100% complete, always |
| **Failure Mode** | Graceful degradation (log warning, continue) | Crash on anomaly |
| **Use Case** | Dashboards, alerting, debugging | Audits, lineage queries, compliance |
| **Source of Truth** | "What is happening right now" | "What happened" |

**Key Principle:** Telemetry complements but never replaces Landscape. If telemetry fails, the pipeline continues. If Landscape fails, the pipeline crashes.

### Two-Path Emission Model

```
PATH 1: ENGINE EVENTS
  Orchestrator --> RunStarted, RunFinished, PhaseChanged, RowCreated
  Processor    --> TransformCompleted, GateEvaluated, TokenCompleted
  Purpose: DAG lifecycle visibility (when rows enter, when transforms run)

PATH 2: CLIENT EVENTS
  AuditedHTTPClient --> ExternalCallCompleted (HTTP)
  AuditedLLMClient  --> ExternalCallCompleted (LLM)
  ctx.record_call() --> ExternalCallCompleted (SQL, Filesystem)
  Purpose: External dependency health (latency, errors, rate limits)
```

**Architectural Decision:** These paths remain separate. Unifying them would conflate concerns and make both less useful.

### Module Structure

```
src/elspeth/telemetry/
├── __init__.py              # Public API exports
├── manager.py               # TelemetryManager - core orchestrator
├── events.py                # Event dataclasses (RunStarted, TransformCompleted, etc.)
├── filtering.py             # Granularity-based filtering (should_emit)
├── buffer.py                # BoundedBuffer for event batching (available, not used)
├── protocols.py             # ExporterProtocol definition
├── errors.py                # TelemetryExporterError exception
├── hookspecs.py             # pluggy hook specs for exporter discovery
└── exporters/
    ├── __init__.py          # BuiltinExportersPlugin registration
    ├── console.py           # ConsoleExporter (stdout/stderr, JSON/pretty)
    ├── otlp.py              # OTLPExporter (OpenTelemetry Protocol via gRPC)
    ├── azure_monitor.py     # AzureMonitorExporter (Application Insights)
    └── datadog.py           # DatadogExporter (Datadog APM via ddtrace)
```

### Thread Safety Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Pipeline Thread                                 │
│                                                                             │
│  Orchestrator ──> TelemetryManager.handle_event() ──> Thread-Safe Queue    │
│  Processor    ──>                                                           │
│  PluginContext──>                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Export Thread                                   │
│                                                                             │
│  Queue ──> Granularity Filter ──> Exporter Dispatch                        │
│                                         │                                   │
│                    ┌────────────────────┼────────────────────┐             │
│                    v                    v                    v              │
│             ┌──────────┐         ┌──────────┐         ┌──────────┐         │
│             │ Console  │         │   OTLP   │         │ Datadog  │         │
│             │ Exporter │         │ Exporter │         │ Exporter │         │
│             └──────────┘         └──────────┘         └──────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

Pipeline thread enqueues events (non-blocking). Export thread consumes and dispatches (background). Only `_events_dropped` counter requires locking (both threads write).

---

## Event Types

### Defined in `telemetry/events.py`

| Event | Granularity | Purpose |
|-------|-------------|---------|
| `RunStarted` | `lifecycle` | Pipeline run begins |
| `RunFinished` | `lifecycle` | Pipeline completes (success/failure) |
| `PhaseChanged` | `lifecycle` | Phase transitions (CONFIG -> GRAPH -> SOURCE -> PROCESS -> EXPORT) |
| `RowCreated` | `rows` | Row enters pipeline from source |

### Defined in `contracts/events.py`

| Event | Granularity | Purpose |
|-------|-------------|---------|
| `TransformCompleted` | `rows` | Transform execution finishes |
| `GateEvaluated` | `rows` | Gate makes routing decision |
| `TokenCompleted` | `rows` | Token reaches terminal state |
| `ExternalCallCompleted` | `full` | HTTP/LLM/SQL call completes |

### Granularity Levels

| Level | Events Included | Typical Volume | Use Case |
|-------|-----------------|----------------|----------|
| `lifecycle` | Run start/finish, phase transitions | ~10-20 per run | Production (minimal overhead) |
| `rows` | Above + row creation, transforms, gates, outcomes | N x M (rows x transforms) | Production (standard) |
| `full` | Above + external call details | High (all LLM/HTTP/SQL calls) | Development, debugging |

---

## Exporters

### Console Exporter
- **Target:** stdout or stderr
- **Formats:** JSON (line-delimited) or pretty (human-readable)
- **Use Case:** Local testing, debugging

```yaml
- name: console
  options:
    format: pretty  # json | pretty
    output: stderr  # stdout | stderr
```

### OTLP Exporter
- **Target:** OpenTelemetry Protocol backends (Jaeger, Tempo, Honeycomb, Grafana Cloud)
- **Transport:** gRPC
- **Features:** Batches events (configurable `batch_size`, default 100), derives `trace_id` from `run_id` (consistent within run), derives `span_id` from `token_id`/`state_id`/`row_id` or timestamp+event_type

```yaml
- name: otlp
  options:
    endpoint: http://localhost:4317
    headers:
      Authorization: "Bearer ${OTEL_TOKEN}"
    batch_size: 100
```

### Azure Monitor Exporter
- **Target:** Azure Application Insights / Azure Monitor
- **Transport:** Native Azure Monitor SDK
- **Features:** Reuses OTLP's span conversion for consistency, adds Azure-specific attributes (`cloud.provider=azure`)

```yaml
- name: azure_monitor
  options:
    connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
    batch_size: 100
```

### Datadog Exporter
- **Target:** Datadog APM
- **Transport:** Native ddtrace library
- **Features:** Creates real Datadog spans with explicit timestamps, all event fields become `elspeth.*` span tags, flattens nested dicts to dotted keys

```yaml
- name: datadog
  options:
    service_name: "elspeth"
    env: "production"
    agent_host: "localhost"
    agent_port: 8126
```

---

## Emission Points

### Orchestrator (`engine/orchestrator/core.py`)

8 emission points - All lifecycle and row-level events:

| Event | Trigger Condition |
|-------|-------------------|
| `RunStarted` | After `recorder.begin_run()` succeeds |
| `RunFinished` | After successful `recorder.finalize_run(COMPLETED)` |
| `PhaseChanged` | GRAPH, SOURCE, PROCESS, EXPORT phase entries |
| `RowCreated` | After creating token for each source row |
| `RunFinished` | After exception caught, `recorder.finalize_run(FAILED)` |

Pattern:
```python
def _emit_telemetry(self, event: TelemetryEvent) -> None:
    if self._telemetry is not None:
        self._telemetry.handle_event(event)
```

### Processor (`engine/processor.py`)

11 emission points - All row-level transform/gate/outcome events:

| Event | Trigger Condition |
|-------|-------------------|
| `TransformCompleted` | After transform execution with retry |
| `GateEvaluated` | After config gate execution |
| `TokenCompleted` | Token routed to sink / gate routing failure / batch aggregation / aggregation error / transform failure / quarantined |

Helper methods: `_emit_transform_completed()`, `_emit_gate_evaluated()`, `_emit_token_completed()`

### Audited Clients

**AuditedHTTPClient** (`plugins/clients/http.py`) - 2 emission points:
- `ExternalCallCompleted` after HTTP request succeeds
- `ExternalCallCompleted` after HTTP request fails

**AuditedLLMClient** (`plugins/clients/llm.py`) - 2 emission points:
- `ExternalCallCompleted` after LLM chat completion succeeds
- `ExternalCallCompleted` after LLM chat completion fails

### PluginContext (`plugins/context.py`)

`record_call()` provides dual recording:
1. Records to Landscape audit trail (legal record)
2. Emits `ExternalCallCompleted` telemetry (operational visibility)

XOR enforcement: exactly ONE of `state_id` or `operation_id` must be set.

**Key Invariant:** Landscape recording happens FIRST, telemetry AFTER. Landscape failure crashes; telemetry failure logs warning and continues.

### Plugin Coverage

#### Source Plugins

| Plugin | External I/O | Telemetry | Status |
|--------|--------------|-----------|--------|
| CSVSource | File read | None | OK (local I/O) |
| JSONSource | File read | None | OK (local I/O) |
| NullSource | None | N/A | OK |
| AzureBlobSource | Azure Blob download | Via `record_call()` | OK |

#### Transform Plugins

| Plugin | External I/O | Telemetry | Status |
|--------|--------------|-----------|--------|
| PassThrough, BatchReplicate, BatchStats, KeywordFilter, Truncate, JSONExplode, FieldMapper | None | N/A | OK |
| AzureContentSafety | Azure API | Via `AuditedHTTPClient` | OK |
| AzurePromptShield | Azure API | Via `AuditedHTTPClient` | OK |

#### Sink Plugins

| Plugin | External I/O | Telemetry | Status |
|--------|--------------|-----------|--------|
| CSVSink | File write | None | OK (local I/O) |
| JSONSink | File write | None | OK (local I/O) |
| DatabaseSink | SQL INSERT | Via `record_call()` | OK |

#### LLM Plugins

| Plugin | External I/O | Telemetry | Status |
|--------|--------------|-----------|--------|
| AzureLLMTransform | Azure OpenAI | Via `AuditedLLMClient` | OK |
| AzureMultiQueryLLMTransform | Azure OpenAI | Via `AuditedLLMClient` | OK |
| OpenRouterLLMTransform | OpenRouter API | Via `AuditedHTTPClient` | OK |
| OpenRouterMultiQueryLLMTransform | OpenRouter API | Via `AuditedHTTPClient` | OK |
| AzureBatchLLMTransform | Azure Batch API | Has `record_call()` but gap | **GAP** |
| OpenRouterBatchLLMTransform | OpenRouter API | Raw `httpx.Client()` | **GAP** |

### Components Without Telemetry (Correct)

| Subsystem | Reason |
|-----------|--------|
| `core/landscape/` | IS the audit trail; telemetry is separate concern |
| `core/checkpoint/` | Internal persistence; no external I/O |
| `core/rate_limit/` | Internal coordination; no external I/O |
| `core/retention/` | Background cleanup; no real-time visibility needed |
| `core/security/` | HMAC fingerprinting; no external I/O |
| `core/canonical.py` | Pure functions; no I/O |
| `core/dag/` | Graph construction; no I/O |
| `cli.py`, `tui/` | User interface; not pipeline execution |

---

## Configuration

### Settings Layer (Pydantic Validation)

```yaml
telemetry:
  enabled: true
  granularity: rows              # lifecycle | rows | full
  backpressure_mode: block       # block | drop
  fail_on_total_exporter_failure: false
  exporters:
    - name: console
      options:
        format: pretty
    - name: otlp
      options:
        endpoint: ${OTEL_ENDPOINT}
```

### Runtime Layer (Frozen Dataclass)

```python
@dataclass(frozen=True, slots=True)
class RuntimeTelemetryConfig:
    enabled: bool
    granularity: TelemetryGranularity  # Enum
    backpressure_mode: BackpressureMode  # Enum
    fail_on_total_exporter_failure: bool
    exporter_configs: tuple[ExporterConfig, ...]
```

Mapping: Settings -> Runtime via `RuntimeTelemetryConfig.from_settings()` with enum conversion and validation.

### Backpressure Modes

| Mode | Behavior | Trade-off | Status |
|------|----------|-----------|--------|
| `block` | Pause pipeline until exporters catch up | Complete telemetry, may slow pipeline | Implemented |
| `drop` | Drop oldest events when queue full | Fast pipeline, lossy telemetry | Implemented |
| `slow` | Adaptive rate limiting | Balanced | Not implemented |

### Example Configurations

**Production:**
```yaml
telemetry:
  enabled: true
  granularity: rows
  backpressure_mode: drop
  fail_on_total_exporter_failure: false
  exporters:
    - name: otlp
      options:
        endpoint: ${OTEL_ENDPOINT}
        batch_size: 100
    - name: datadog
      options:
        service_name: "elspeth-pipeline"
        env: "production"
```

**Debugging:**
```yaml
telemetry:
  enabled: true
  granularity: full
  backpressure_mode: block
  exporters:
    - name: console
      options:
        format: pretty
        output: stderr
```

---

## Failure Handling

### Design Principles

1. **Telemetry Never Crashes Pipeline:** All telemetry failures are caught and logged
2. **Aggregate Logging:** Warnings every 100 failures (prevents log spam)
3. **Failure Isolation:** Individual exporter failures don't affect others
4. **Configurable Total Failure:** `fail_on_total_exporter_failure` (default: false)

### Health Metrics

```python
{
    "events_emitted": int,           # Successfully exported
    "events_dropped": int,           # Queue full or all exporters failed
    "exporter_failures": {name: count},
    "consecutive_total_failures": int,
    "queue_depth": int,
    "queue_maxsize": int,
}
```

---

## Correlation Workflow

Telemetry events include `run_id` and `token_id` for correlation:

```python
def _derive_trace_id(run_id: str) -> int:
    """All events from same run get same trace_id."""
    return int(hashlib.sha256(run_id.encode()).hexdigest()[:16], 16)
```

**Workflow:**
1. Alert fires in Datadog/Grafana for high latency
2. Extract `run_id` from telemetry span
3. Use `elspeth explain --run <run_id> --database <path/to/audit.db>` or Landscape MCP server to investigate full lineage

---

## Remediation: Root Cause Analysis

### The Misdiagnosis

The original telemetry audit concluded "6 plugins emit telemetry correctly." This was **incorrect** - the audit analyzed plugins in isolation, examining whether they *use* the telemetry callback, but failed to verify whether the callback is ever *wired* in production.

### The Actual Bug

**Location:** `src/elspeth/engine/orchestrator/core.py`

PluginContext was created without wiring `telemetry_emit`, which defaults to `lambda event: None`. This meant ALL plugins silently dropped telemetry. The Landscape audit trail worked (separate code path); only telemetry was broken.

**Why tests didn't catch it:** Unit tests create PluginContext manually with telemetry wired. Production uses Orchestrator, which didn't wire it. Classic test/production code path divergence.

### Remediation Phases

**Phase 1 (Critical):** Wire `telemetry_emit` in Orchestrator
```python
telemetry_emit=self._emit_telemetry,  # Add to PluginContext creation
```

**Phase 2 (High):** Fix operation context bug - `record_call()` uses `state_id=operation_id` for source/sink context, breaking correlation with `node_states` table.

**Phase 3 (Medium):** Aggregation flush missing telemetry - `execute_flush()` doesn't emit `TransformCompleted`.

**Phase 4 (Medium):** OpenRouterBatchLLMTransform uses raw `httpx.Client()` instead of `AuditedHTTPClient` - complete blind spot even after orchestrator fix.

### Decisions Made

| Decision | Outcome | Rationale |
|----------|---------|-----------|
| File I/O telemetry | REJECTED | Volume explosion, no operational value for local I/O |
| Two-path emission model | CONFIRMED | Separate concerns (DAG lifecycle vs dependency health) |
| `track_operation` telemetry | REJECTED | Violates separation of concerns |

---

## Implementation Status

### Complete

| Component | Notes |
|-----------|-------|
| TelemetryManager | Thread-safe async export, backpressure, failure isolation |
| Console Exporter | JSON and pretty formats |
| OTLP Exporter | gRPC transport, batch export |
| Azure Monitor Exporter | Native SDK integration |
| Datadog Exporter | ddtrace integration |
| All Event Types | Lifecycle, row-level, external call events |
| Configuration | Settings -> Runtime with protocol enforcement |
| Granularity Filtering | lifecycle/rows/full levels |
| Backpressure (block/drop) | Queue-based with configurable mode |

### Gaps

| Gap | Priority | Notes |
|-----|----------|-------|
| Orchestrator telemetry_emit wiring | P0 | Root cause - all plugin telemetry broken |
| Operation context ID bug | P1 | state_id field contains operation_id for source/sink |
| OpenRouterBatchLLMTransform | P1 | Uses raw httpx.Client, no instrumentation |
| AzureBatchLLMTransform | P2 | record_call works but operation context bug affects it |
| Aggregation flush telemetry | P3 | execute_flush() doesn't emit TransformCompleted |
| OTLP time-based flushing | P4 | flush_interval_ms accepted but not implemented |
| Slow backpressure mode | P4 | Placeholder only |

---

## Key Files

| File | Purpose |
|------|---------|
| `engine/orchestrator/core.py` | Lifecycle events, PluginContext wiring |
| `engine/processor.py` | Row-level events (TransformCompleted, GateEvaluated, TokenCompleted) |
| `plugins/clients/http.py` | AuditedHTTPClient (ExternalCallCompleted for HTTP) |
| `plugins/clients/llm.py` | AuditedLLMClient (ExternalCallCompleted for LLM) |
| `plugins/context.py` | PluginContext, record_call() unified recording + telemetry |
| `telemetry/manager.py` | TelemetryManager queue management and dispatch |
| `telemetry/events.py` | Event definitions |
| `telemetry/filtering.py` | Granularity-based filtering |
| `telemetry/exporters/` | Console, OTLP, Azure Monitor, Datadog exporters |
| `contracts/events.py` | TransformCompleted, GateEvaluated, TokenCompleted definitions |

### Quick Reference: Find Emission Points

```bash
# All telemetry_emit calls
rg "telemetry_emit\(" src/elspeth/

# All TelemetryEvent instantiation
rg "(RunStarted|RunFinished|PhaseChanged|RowCreated|ExternalCallCompleted|TransformCompleted|GateEvaluated|TokenCompleted)\(" src/elspeth/

# All handle_event calls
rg "handle_event\(" src/elspeth/
```
