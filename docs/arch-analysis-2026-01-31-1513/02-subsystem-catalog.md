## Subsystem Catalog (Audit + Telemetry Focus)

### 1) Landscape Audit Core
- **Location:** `src/elspeth/core/landscape/`
- **Responsibility:** Authoritative audit trail (runs, nodes, rows, tokens, node_states, external calls, artifacts). Handles hashing, payload references, schema integrity, export.
- **Key components:**
  - `recorder.py` (LandscapeRecorder API)
  - `schema.py` (SQLAlchemy tables)
  - `repositories.py` (row → model mapping)
  - `database.py` (DB engine + setup)
  - `exporter.py` (audit export)
- **Inbound dependencies:** Engine (`engine/orchestrator.py`, `engine/processor.py`, `engine/executors.py`), TokenManager (`engine/tokens.py`), PluginContext (`plugins/context.py`), Audited clients (`plugins/clients/*`).
- **Outbound dependencies:** Canonicalization (`core/canonical.py`), Payload store (`core/payload_store.py`), Contracts (`contracts/*`).
- **Patterns observed:**
  - Recorder facade around SQLAlchemy Core
  - Strict hashing + optional payload storage
  - Call index allocation centralized (recorder)
- **Confidence:** High

### 2) Telemetry Core
- **Location:** `src/elspeth/telemetry/`
- **Responsibility:** Operational telemetry event definitions, filtering by granularity, async export pipeline, exporter registry.
- **Key components:**
  - `events.py` (RunStarted, RowCreated, ExternalCallCompleted, etc.)
  - `manager.py` (TelemetryManager, queue, exporter dispatch)
  - `filtering.py` (granularity filter)
  - `exporters/*` (console, OTLP, Azure Monitor, Datadog)
- **Inbound dependencies:** Engine (orchestrator + processor), PluginContext and Audited clients (external call telemetry).
- **Outbound dependencies:** Exporter SDKs (ddtrace, opentelemetry), structlog.
- **Patterns observed:**
  - Async export thread
  - Fail‑open/disable on repeated exporter failure
  - Granularity filtering (lifecycle/rows/full)
- **Confidence:** High

### 3) Engine Emission Layer
- **Location:** `src/elspeth/engine/`
- **Responsibility:** Pipeline execution + audit/telemetry emission at run, row, node, gate, sink levels.
- **Key components:**
  - `orchestrator.py` (run lifecycle, source iteration, RowCreated telemetry)
  - `processor.py` (TransformCompleted/GateEvaluated/TokenCompleted telemetry)
  - `executors.py` (node_state lifecycle for transforms/sinks/gates)
  - `tokens.py` (row/token creation in Landscape)
- **Inbound dependencies:** CLI / programmatic callers.
- **Outbound dependencies:** LandscapeRecorder, TelemetryManager, PluginContext.
- **Patterns observed:**
  - “Landscape first, telemetry after” ordering
  - Node_state is the audit spine for transforms/sinks
- **Confidence:** High

### 4) Plugin Integration Layer (Context + Audited Clients)
- **Location:** `src/elspeth/plugins/context.py`, `src/elspeth/plugins/clients/*`
- **Responsibility:** Provide plugins with audit/telemetry hooks; audited HTTP/LLM clients record external calls.
- **Key components:**
  - `PluginContext.record_call()` (records to Landscape, emits ExternalCallCompleted)
  - `AuditedLLMClient` / `AuditedHTTPClient` (automatic audit + telemetry)
- **Inbound dependencies:** Engine sets context fields (run_id, landscape, state_id, operation_id, telemetry_emit).
- **Outbound dependencies:** LandscapeRecorder, Telemetry events.
- **Patterns observed:**
  - Telemetry callback always present (no‑op when disabled)
  - External call recording centralized
- **Confidence:** High

### 5) Sources (CSV/JSON/Null)
- **Location:** `src/elspeth/plugins/sources/*`
- **Responsibility:** Ingest external data, validate/coerce, quarantine failures, emit rows.
- **Key components:** `csv_source.py`, `json_source.py`, `null_source.py`
- **Inbound dependencies:** Orchestrator + PluginContext.
- **Outbound dependencies:** PluginContext.record_validation_error(), TokenManager.create_initial_token() → LandscapeRecorder.create_row/create_token().
- **Patterns observed:**
  - Validation errors recorded to Landscape via ctx
  - Source load wrapped in `track_operation` (operations API)
- **Confidence:** Medium (no API/db sources in this repo)

### 6) Sinks (CSV/JSON/Database)
- **Location:** `src/elspeth/plugins/sinks/*`
- **Responsibility:** Persist output, return artifact descriptors.
- **Key components:** `csv_sink.py`, `json_sink.py`, `database_sink.py`
- **Inbound dependencies:** SinkExecutor (node_state + artifact recording), PluginContext (state_id set for external call attribution).
- **Outbound dependencies:** LandscapeRecorder.complete_node_state(), register_artifact(), record_token_outcome().
- **Patterns observed:**
  - Node_state per token; artifact registered post‑flush
  - No explicit sink‑level telemetry event
- **Confidence:** Medium

### 7) LLM Transforms
- **Location:** `src/elspeth/plugins/llm/*`
- **Responsibility:** External LLM calls with audit recording + telemetry for external calls.
- **Key components:** `azure.py`, `openrouter.py`, `azure_multi_query.py`, `openrouter_multi_query.py`, `base.py`
- **Inbound dependencies:** TransformExecutor sets ctx.state_id, ctx.telemetry_emit.
- **Outbound dependencies:** AuditedLLMClient → LandscapeRecorder.record_call() + telemetry ExternalCallCompleted.
- **Patterns observed:**
  - “Landscape then telemetry” for each call
  - Response payload recorded (including raw response) for audit completeness
- **Confidence:** High
