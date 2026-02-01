# Telemetry Implementation Summary

> **Status:** RC-2 (Largely Complete)
> **Last Updated:** 2026-01-31
> **Purpose:** Operational visibility into pipeline execution - real-time monitoring, not audit trail

## What Telemetry Is (And Isn't)

| Aspect | Telemetry | Landscape (Audit Trail) |
|--------|-----------|------------------------|
| **Purpose** | Real-time operational visibility | Legal record of what happened |
| **Persistence** | Ephemeral (streaming to external systems) | Permanent (SQLite/PostgreSQL) |
| **Completeness** | Configurable (granularity levels) | 100% complete, always |
| **Failure Mode** | Graceful degradation (log warning, continue) | Crash on anomaly |
| **Use Case** | Dashboards, alerting, debugging | Audits, lineage queries, compliance |
| **Source of Truth** | "What is happening right now" | "What happened" |

**Key Principle:** Telemetry complements but never replaces Landscape. If telemetry fails, the pipeline continues. If Landscape fails, the pipeline crashes.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Pipeline Thread                                 │
│                                                                             │
│  Orchestrator ──► TelemetryManager.handle_event() ──► Thread-Safe Queue    │
│  Processor    ──►                                                           │
│  PluginContext──►                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Export Thread                                   │
│                                                                             │
│  Queue ──► Exporter Dispatch ──► Console / OTLP / Azure Monitor / Datadog  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Thread Safety:** Pipeline thread enqueues events (non-blocking). Export thread consumes and dispatches (background). Only `_events_dropped` counter requires locking (both threads write).

---

## Module Structure

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

---

## Implemented Exporters

### Console Exporter
- **Target:** stdout or stderr
- **Formats:** JSON (line-delimited) or pretty (human-readable)
- **Use Case:** Local testing, debugging
- **Status:** ✅ Complete

```yaml
- name: console
  options:
    format: pretty  # json | pretty
    output: stderr  # stdout | stderr
```

### OTLP Exporter
- **Target:** OpenTelemetry Protocol backends (Jaeger, Tempo, Honeycomb, Grafana Cloud)
- **Transport:** gRPC
- **Features:**
  - Batches events (configurable `batch_size`, default 100)
  - Derives `trace_id` from `run_id` (consistent within run)
  - Derives `span_id` from `token_id`/`state_id`/`row_id` or timestamp+event_type
  - Converts all event fields to span attributes
- **Status:** ✅ Complete

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
- **Features:**
  - Reuses OTLP's span conversion for consistency
  - Adds Azure-specific attributes (`cloud.provider=azure`)
  - Batches events (configurable `batch_size`, default 100)
- **Status:** ✅ Complete

```yaml
- name: azure_monitor
  options:
    connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
    batch_size: 100
```

### Datadog Exporter
- **Target:** Datadog APM
- **Transport:** Native ddtrace library
- **Features:**
  - Creates real Datadog spans with explicit timestamps
  - All event fields become `elspeth.*` span tags
  - Flattens nested dicts to dotted keys (e.g., `token_usage.prompt_tokens`)
  - Supports local Datadog Agent or direct API
- **Status:** ✅ Complete

```yaml
- name: datadog
  options:
    api_key: ${DD_API_KEY}          # Optional with local agent
    service_name: "elspeth"
    env: "production"
    agent_host: "localhost"
    agent_port: 8126
```

---

## Event Types

### Lifecycle Events (Always Emitted)

| Event | Fields | Emitted From |
|-------|--------|--------------|
| `RunStarted` | `run_id`, `config_hash`, `source_plugin` | Orchestrator |
| `RunFinished` | `run_id`, `status`, `row_count`, `duration_ms` | Orchestrator |
| `PhaseChanged` | `run_id`, `phase`, `action` | Orchestrator |

### Row-Level Events (Granularity: `rows` or `full`)

| Event | Fields | Emitted From |
|-------|--------|--------------|
| `RowCreated` | `run_id`, `row_id`, `token_id`, `content_hash` | Orchestrator |
| `TransformCompleted` | `run_id`, `row_id`, `token_id`, `node_id`, `plugin_name`, `status`, `duration_ms`, `input_hash`, `output_hash` | Processor |
| `GateEvaluated` | `run_id`, `row_id`, `token_id`, `node_id`, `plugin_name`, `routing_mode`, `destinations` | Processor |
| `TokenCompleted` | `run_id`, `row_id`, `token_id`, `outcome`, `sink_name` | Processor |

### External Call Events (Granularity: `full` only)

| Event | Fields | Emitted From |
|-------|--------|--------------|
| `ExternalCallCompleted` | `run_id`, `state_id`, `call_type`, `provider`, `status`, `latency_ms`, `request_hash`, `response_hash`, `token_usage` | PluginContext |

---

## Granularity Levels

| Level | Events Included | Typical Volume | Use Case |
|-------|-----------------|----------------|----------|
| `lifecycle` | Run start/finish, phase transitions | ~10-20 per run | Production (minimal overhead) |
| `rows` | Above + row creation, transforms, gates, outcomes | N × M (rows × transforms) | Production (standard) |
| `full` | Above + external call details | High (all LLM/HTTP/SQL calls) | Development, debugging |

---

## Backpressure Modes

| Mode | Behavior | Trade-off | Status |
|------|----------|-----------|--------|
| `block` | Pause pipeline until exporters catch up | Complete telemetry, may slow pipeline | ✅ Implemented |
| `drop` | Drop oldest events when queue full | Fast pipeline, lossy telemetry | ✅ Implemented |
| `slow` | Adaptive rate limiting | Balanced | ❌ Not implemented |

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

**Mapping:** Settings → Runtime via `RuntimeTelemetryConfig.from_settings()` with enum conversion and validation.

---

## Engine Integration

### Orchestrator (`src/elspeth/engine/orchestrator.py`)

```python
def _emit_telemetry(self, event: TelemetryEvent) -> None:
    """Emit telemetry event if manager is configured."""
    if self._telemetry is not None:
        self._telemetry.handle_event(event)
```

**Emission Points:**
- `RunStarted` after run_id generated
- `PhaseChanged` for each phase transition
- `RowCreated` after landscape recording succeeds
- `RunFinished` at run completion

### Processor (`src/elspeth/engine/processor.py`)

**Emission Points:**
- `TransformCompleted` after transform execution
- `GateEvaluated` after gate routing
- `TokenCompleted` when token reaches terminal state

### Plugin Context (`src/elspeth/plugins/context.py`)

```python
def record_call(self, ...) -> Call | None:
    """Record external call and emit telemetry."""
    recorded_call = self.landscape.record_call(...)  # Landscape FIRST

    try:
        self.telemetry_emit(ExternalCallCompleted(...))  # Telemetry AFTER
    except Exception as e:
        logger.warning("telemetry_emit_failed", error=str(e))

    return recorded_call
```

**Key Invariant:** Landscape recording happens FIRST, telemetry AFTER. Landscape failure crashes; telemetry failure logs warning and continues.

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

## Implementation Status

### Complete ✅

| Component | Notes |
|-----------|-------|
| TelemetryManager | Thread-safe async export, backpressure, failure isolation |
| Console Exporter | JSON and pretty formats |
| OTLP Exporter | gRPC transport, batch export |
| Azure Monitor Exporter | Native SDK integration |
| Datadog Exporter | ddtrace integration |
| Lifecycle Events | RunStarted, RunFinished, PhaseChanged |
| Row-Level Events | RowCreated, TransformCompleted, GateEvaluated, TokenCompleted |
| External Call Events | ExternalCallCompleted via PluginContext |
| Configuration | Settings → Runtime with protocol enforcement |
| Granularity Filtering | lifecycle/rows/full levels |
| Backpressure (block/drop) | Queue-based with configurable mode |

### Incomplete/Rough ❌

| Gap | Priority | Notes |
|-----|----------|-------|
| Aggregation flush telemetry | P3 | `execute_flush()` doesn't emit `TransformCompleted` |
| Source/Sink external I/O telemetry | P1 | Azure Blob, Database sinks missing telemetry emission |
| OTLP time-based flushing | P4 | `flush_interval_ms` accepted but not implemented |
| Slow backpressure mode | P4 | Placeholder only |

### Known Issues

**P3-2026-01-31-aggregation-flush-missing-telemetry:**
- Location: `src/elspeth/engine/processor.py:480-519`
- Issue: `execute_flush()` returns aggregated rows but doesn't emit telemetry
- Fix: Call `_emit_transform_completed()` after flush

**P1-Source/Sink Telemetry (from audit):**
- Plugins with external I/O need `telemetry_emit` calls:
  - `AzureBlobSource` - Blob downloads
  - `AzureBlobSink` - Blob uploads
  - `DatabaseSink` - SQL inserts
- Local file I/O (CSVSource, JSONSource) less critical (sync, fast, local)

---

## Testing

| Area | Coverage |
|------|----------|
| Manager (thread, backpressure, failures) | Good |
| Exporters (all four) | Good |
| Events filtering (granularity) | Good |
| Config (settings → runtime) | Good |
| Orchestrator integration | Partial |
| Processor integration | Partial (excludes aggregations) |
| Plugin context | Good |

**Test Files:**
- `tests/engine/test_orchestrator_telemetry.py`
- `tests/engine/test_processor_telemetry.py`
- `tests/contracts/test_telemetry_config.py`
- `tests/telemetry/test_*.py`

---

## Quick Reference

### Enable Telemetry

```yaml
telemetry:
  enabled: true
  granularity: rows
  backpressure_mode: block
  exporters:
    - name: console
      options:
        format: pretty
```

### Production Configuration

```yaml
telemetry:
  enabled: true
  granularity: rows
  backpressure_mode: drop  # Prefer pipeline speed over telemetry completeness
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

### Debugging Configuration

```yaml
telemetry:
  enabled: true
  granularity: full  # Include external call details
  backpressure_mode: block  # Don't drop events
  exporters:
    - name: console
      options:
        format: pretty
        output: stderr
```
