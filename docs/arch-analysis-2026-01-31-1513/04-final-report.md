# Audit + Telemetry Deep Dive (Sources/Sinks/LLM)

## Executive Summary
- **Landscape audit trail** is well‑implemented and is the authoritative path for run, node, row/token, node_state, and external call recording.
- **Telemetry subsystem exists** (events, manager, exporters) but **is not wired into the CLI/orchestrator execution path**, resulting in effectively **zero telemetry emission** in the default run path.
- **Sources and LLM transforms** do record audit data (rows, tokens, calls), but **sinks and source/sink operations do not emit telemetry or operation‑scoped call records**, leaving gaps vs. the stated “every external call recorded” principle.

## Top‑Down: Landscape (Audit) System
**Core path:** `LandscapeRecorder` is the primary API for audit writes (runs, nodes, rows, tokens, node_states, calls, artifacts). It owns hashing and payload storage and is invoked from the engine and plugin layer.

- **Run and graph metadata**: `Orchestrator` registers nodes and edges in Landscape before processing rows. (`src/elspeth/engine/orchestrator.py`)
- **Source row entry**: Source rows are recorded via `TokenManager.create_initial_token()` → `LandscapeRecorder.create_row()` and `create_token()` with optional payload storage. (`src/elspeth/engine/tokens.py`, `src/elspeth/core/landscape/recorder.py`)
- **Transform/gate/sink processing**: Executors create `node_states` per token and complete them with output, then record terminal outcomes. (`src/elspeth/engine/executors.py`)
- **External calls**: `LandscapeRecorder.record_call()` and `record_operation_call()` store request/response hashes and payload references. (`src/elspeth/core/landscape/recorder.py`)

## Top‑Down: Telemetry System
**Core path:** Telemetry defines event types, filters them by granularity, and ships them through exporters via `TelemetryManager`.

- **Events**: Lifecycle (`RunStarted`, `RunFinished`, `PhaseChanged`), row‑level (`RowCreated`, `TransformCompleted`, `GateEvaluated`, `TokenCompleted`), external calls (`ExternalCallCompleted`). (`src/elspeth/telemetry/events.py`, `src/elspeth/contracts/events.py`)
- **Manager**: `TelemetryManager` handles queueing, filtering, exporter dispatch, and backpressure logic. (`src/elspeth/telemetry/manager.py`)
- **Exporters**: Console, OTLP, Azure Monitor, Datadog. (`src/elspeth/telemetry/exporters/*`)

## Trace Map: How Data Flows Into Audit + Telemetry

| Component | Audit Trail (Landscape) | Telemetry | Notes |
|---|---|---|---|
| **Orchestrator** | `begin_run`, `register_node`, `register_edge`, `finalize_run` | `RunStarted`, `RunFinished`, `PhaseChanged` (if wired) | `src/elspeth/engine/orchestrator.py` |
| **RowProcessor** | n/a (uses recorder via executors) | `TransformCompleted`, `GateEvaluated`, `TokenCompleted` (if wired) | `src/elspeth/engine/processor.py` |
| **Sources (CSV/JSON)** | `record_validation_error`, row/token creation | `RowCreated` emitted after row creation (if wired) | `src/elspeth/plugins/sources/*`, `engine/orchestrator.py` |
| **LLM transforms** | `record_call` (full request/response) via `AuditedLLMClient` | `ExternalCallCompleted` (if wired) | `src/elspeth/plugins/llm/*`, `plugins/clients/llm.py` |
| **Sinks (CSV/JSON/DB)** | `begin_node_state`/`complete_node_state`, `register_artifact`, `record_token_outcome` | No sink‑specific event | `src/elspeth/engine/executors.py`, `plugins/sinks/*` |
| **PluginContext.record_call** | `record_call` or `record_operation_call` | `ExternalCallCompleted` (if wired) | `src/elspeth/plugins/context.py` |

## Findings: Audit/Telemetry Gaps (Observed in Code)

### 1) TelemetryManager is not wired into the CLI execution path
- **Evidence:** `Orchestrator` is constructed in `src/elspeth/cli.py` without a `telemetry_manager` argument; `RowProcessor` is instantiated in `src/elspeth/engine/orchestrator.py` without passing `telemetry_manager`; `PluginContext.telemetry_emit` is never set. The telemetry subsystem exists but is effectively unused.
- **Impact:** No telemetry events (lifecycle, row‑level, external calls) are emitted in default runs.

### 2) PluginContext.telemetry_emit remains default no‑op
- **Evidence:** `PluginContext` defaults `telemetry_emit` to a no‑op function (`src/elspeth/plugins/context.py`). Orchestrator creates `PluginContext` without setting this callback (`src/elspeth/engine/orchestrator.py`).
- **Impact:** Audited clients and `ctx.record_call()` always “emit” into a no‑op; `ExternalCallCompleted` telemetry is dropped silently.

### 3) Sink operations are not tracked with `track_operation`
- **Evidence:** `track_operation` is only used around source load (`src/elspeth/engine/orchestrator.py`); no sink write path uses it. `record_operation_call()` is only reachable via `ctx.record_call()` when `ctx.operation_id` is set, which never occurs for sinks.
- **Impact:** No operation‑level audit trail for sink I/O; external calls during sinks can’t be attributed to an operation scope.

### 4) Sinks do not record external calls (SQL/Filesystem) in `calls` table
- **Evidence:** Sink implementations (`src/elspeth/plugins/sinks/*`) do not call `ctx.record_call()` and do not use audited clients. Database sink writes directly via SQLAlchemy without audit call records.
- **Impact:** External calls made by sinks are absent from `calls` table, despite having `CallType.SQL/FILESYSTEM` support in contracts.

### 5) Telemetry “no silent failure” rule not enforced when disabled/no exporters
- **Evidence:** `TelemetryManager.handle_event()` returns silently when `self._exporters` is empty or telemetry is disabled (`src/elspeth/telemetry/manager.py`). CLAUDE.md requires explicit acknowledgment when telemetry is unavailable.
- **Impact:** Operational visibility is silently absent; operators get no signal that telemetry is off.

## Notes on Sources / LLM / Sinks (Specific Focus)

### Sources
- **Audit:** Validation errors recorded via `ctx.record_validation_error()` and rows/tokens created via `TokenManager` and `LandscapeRecorder`. (`src/elspeth/plugins/sources/*`, `src/elspeth/engine/tokens.py`)
- **Telemetry:** `RowCreated` is emitted after row creation in orchestrator, but only if telemetry is wired. (`src/elspeth/engine/orchestrator.py`)

### LLM Plugins
- **Audit:** `AuditedLLMClient` records full request/response (including raw responses and usage) to Landscape. (`src/elspeth/plugins/clients/llm.py`)
- **Telemetry:** `ExternalCallCompleted` is emitted after audit recording, but callback is currently no‑op. (`src/elspeth/plugins/llm/*`, `src/elspeth/plugins/context.py`)

### Sinks
- **Audit:** Node states and artifacts are recorded after sink flush; token outcomes are finalized there. (`src/elspeth/engine/executors.py`)
- **Telemetry:** No sink‑level telemetry event exists; no external call telemetry for sink I/O.

## Validation (Limitations)
- Subagent tooling is unavailable in this environment, so validation is self‑performed. This deviates from the mandated multi‑subsystem validation gate; treat findings as high‑confidence but single‑review.
