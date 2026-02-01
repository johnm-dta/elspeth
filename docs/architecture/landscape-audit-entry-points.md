# Landscape Audit Entry Points

> **Complete Reference: What Records to the Audit Trail and When**

This document maps every component in ELSPETH that feeds audit entries into the Landscape system, organized by execution layer and timing.

---

## Table of Contents

1. [Overview](#overview)
2. [Entry Point Summary](#entry-point-summary)
3. [Engine Layer](#engine-layer)
4. [Plugin & Client Layer](#plugin--client-layer)
5. [Checkpoint Subsystem](#checkpoint-subsystem)
6. [Coalesce & Aggregation Paths](#coalesce--aggregation-paths)
7. [LandscapeRecorder API Reference](#landscaperecorder-api-reference)
8. [Timing Diagrams](#timing-diagrams)

---

## Overview

Audit entries flow into Landscape from **four primary layers**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ENGINE LAYER                                                                │
│  orchestrator.py, processor.py, executors.py, coalesce_executor.py          │
│  • Run lifecycle, DAG registration, token outcomes, routing events          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PLUGIN/CLIENT LAYER                                                         │
│  AuditedHTTPClient, AuditedLLMClient, PluginContext                         │
│  • External calls (HTTP, LLM, SQL), validation/transform errors             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CHECKPOINT SUBSYSTEM                                                        │
│  checkpoint/manager.py, checkpoint/recovery.py                              │
│  • Checkpoint creation, checkpoint deletion, recovery queries               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LANDSCAPE RECORDER                                                          │
│  core/landscape/recorder.py (~2,800 lines, 60+ methods)                     │
│  • All database writes, payload persistence, hash computation               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Entry Point Summary

### By Trigger Event

| Event | Components Recording | Tables Written |
|-------|---------------------|----------------|
| **Run starts** | Orchestrator | runs, nodes, edges |
| **Source yields row** | Orchestrator | rows, tokens |
| **Transform processes** | TransformExecutor | node_states |
| **Gate routes** | GateExecutor | node_states, routing_events |
| **External call made** | AuditedHTTPClient, AuditedLLMClient | calls |
| **Sink writes** | SinkExecutor | node_states, artifacts, token_outcomes |
| **Fork occurs** | TokenManager | tokens, token_parents, token_outcomes |
| **Coalesce merges** | CoalesceExecutor | node_states, token_outcomes |
| **Aggregation buffers** | AggregationExecutor | batches, batch_members |
| **Aggregation flushes** | AggregationExecutor | node_states, batches |
| **Error occurs** | PluginContext | validation_errors, transform_errors |
| **Checkpoint created** | CheckpointManager | checkpoints |
| **Run completes** | Orchestrator | runs |

### By Database Table

| Table | Primary Writers | When |
|-------|-----------------|------|
| `runs` | Orchestrator | Start, status changes, completion |
| `nodes` | Orchestrator | DAG registration (once per run) |
| `edges` | Orchestrator | DAG registration (once per run) |
| `rows` | Orchestrator | Source yields each row |
| `tokens` | Orchestrator, TokenManager | Row creation, fork, coalesce, expand |
| `token_parents` | TokenManager | Fork, coalesce, expand (atomic) |
| `token_outcomes` | Processor, Executors | Terminal states |
| `node_states` | All Executors | Begin/complete processing |
| `routing_events` | GateExecutor | Gate routing decisions |
| `calls` | AuditedClients, PluginContext | External HTTP/LLM/SQL calls |
| `operations` | Orchestrator | Source load, sink write I/O |
| `batches` | AggregationExecutor | Batch lifecycle |
| `batch_members` | AggregationExecutor | Each buffered token |
| `artifacts` | SinkExecutor | Sink output files |
| `validation_errors` | PluginContext | Source validation failures |
| `transform_errors` | PluginContext | Transform processing errors |
| `checkpoints` | CheckpointManager | After sink writes |

---

## Engine Layer

### Orchestrator (`src/elspeth/engine/orchestrator.py`)

The orchestrator controls run lifecycle and coordinates all other components.

#### Run Lifecycle Recording

| Method | Line | Recorder Call | When | What's Recorded |
|--------|------|---------------|------|-----------------|
| `run()` | 565 | `begin_run()` | Run initialization | config, canonical_version, source_schema |
| `run()` | 601 | `finalize_run(COMPLETED)` | Successful completion | Final status, reproducibility grade |
| `run()` | 714 | `finalize_run(FAILED)` | Run failure | Failed status |
| `resume()` | 1940 | `update_run_status(RUNNING)` | Resume starts | Status change |
| `resume()` | 1967, 1995 | `finalize_run(COMPLETED)` | Resume completes | Final status |

#### DAG Registration

| Method | Line | Recorder Call | When | What's Recorded |
|--------|------|---------------|------|-----------------|
| `_register_dag()` | 866 | `register_node()` | Once per node | plugin_name, node_type, config, determinism, schema |
| `_register_dag()` | 882 | `register_edge()` | Once per edge | from_node, to_node, label, routing_mode |

#### Source Row Recording

| Method | Line | Recorder Call | When | What's Recorded |
|--------|------|---------------|------|-----------------|
| `_execute_run()` | 1148 | `record_source_field_resolution()` | First row | Field name normalization mapping |
| `_execute_run()` | 1183 | `record_token_outcome(QUARANTINED)` | Quarantined source row | Error hash, sink name |

#### Export Recording

| Method | Line | Recorder Call | When | What's Recorded |
|--------|------|---------------|------|-----------------|
| `run()` | 624 | `set_export_status(PENDING)` | Export phase starts | Format, target sink |
| `run()` | 651 | `set_export_status(COMPLETED)` | Export succeeds | Completion status |
| `run()` | 655 | `set_export_status(FAILED)` | Export fails | Error message |

---

### RowProcessor (`src/elspeth/engine/processor.py`)

The processor handles DAG traversal and records terminal token outcomes.

#### Token Outcome Recording

| Line | Outcome | When | Context |
|------|---------|------|---------|
| 515 | `FAILED` | Timeout flush fails (passthrough mode) | Buffered tokens need terminal state |
| 817 | `FAILED` | Timeout flush fails (non-passthrough) | Transform failure during timeout |
| 847 | `CONSUMED_IN_BATCH` | Timeout flush partial recovery | Current token only |
| 1034 | `CONSUMED_IN_BATCH` | Aggregation expansion | Triggering token consumed |
| 1098 | `BUFFERED` | Aggregation passthrough buffer | Non-terminal, awaiting flush |
| 1114 | `CONSUMED_IN_BATCH` | Aggregation non-passthrough buffer | Terminal consumption |
| 1516 | `FAILED` | Coalesce failure | Branch requirements not met |
| 1705 | `FAILED` | Max retries exceeded | Transform exhausted retries |
| 1729 | `QUARANTINED` | Transform error discarded | Intentional discard routing |

---

### TransformExecutor (`src/elspeth/engine/executors.py`)

Records node_state lifecycle for transform processing.

| Line | Recorder Call | When | What's Recorded |
|------|---------------|------|-----------------|
| 223 | `begin_node_state()` | Transform starts | token_id, node_id, input_data, attempt |
| 295 | `complete_node_state(FAILED)` | Exception raised | Error details, duration |
| 354 | `complete_node_state(COMPLETED)` | Success | Output data, duration |
| 372 | `complete_node_state(FAILED)` | Error result returned | Transform error details |

---

### GateExecutor (`src/elspeth/engine/executors.py`)

Records node_state and routing decisions.

| Line | Recorder Call | When | What's Recorded |
|------|---------------|------|-----------------|
| 499 | `begin_node_state()` | Gate evaluation starts | token_id, node_id, input_data |
| 526 | `complete_node_state(FAILED)` | Exception raised | Error details |
| 565 | `complete_node_state(FAILED)` | Missing edge | MissingEdgeError |
| 598 | `complete_node_state(FAILED)` | Fork without TokenManager | RuntimeError |
| 625 | `complete_node_state(COMPLETED)` | Gate succeeds | Output data |
| 863 | `record_routing_event()` | Single destination | edge_id, mode, reason |
| 878 | `record_routing_events()` | Fork (multiple destinations) | List of routes, routing_group_id |

---

### SinkExecutor (`src/elspeth/engine/executors.py`)

Records sink operations and terminal outcomes.

| Line | Recorder Call | When | What's Recorded |
|------|---------------|------|-----------------|
| 1668 | `begin_node_state()` | Sink execution starts | Each token individually |
| 1709 | `complete_node_state(FAILED)` | Exception raised | Error for each token |
| 1735 | `complete_node_state(COMPLETED)` | Success | Artifact reference |
| 1745 | `register_artifact()` | After successful write | Path, content_hash |
| 1766 | `record_token_outcome()` | After sink operation | COMPLETED or ROUTED |

---

### AggregationExecutor (`src/elspeth/engine/executors.py`)

Records batch lifecycle and aggregation processing.

#### Buffering Phase

| Line | Recorder Call | When | What's Recorded |
|------|---------------|------|-----------------|
| 965 | `create_batch()` | First row buffered | aggregation_node_id, status=DRAFT |
| 981 | `add_batch_member()` | Each row buffered | batch_id, token_id, ordinal |

#### Flush Phase

| Line | Recorder Call | When | What's Recorded |
|------|---------------|------|-----------------|
| 1089 | `update_batch_status(EXECUTING)` | Flush begins | trigger_type |
| 1102 | `begin_node_state()` | Batch processing starts | Input = `{"batch_rows": [...]}` |
| 1131 | `complete_node_state(PENDING)` | BatchPendingError | Async submission |
| 1139 | `update_batch_status()` | Link batch to state | state_id |
| 1156 | `complete_node_state(FAILED)` | Exception | Error details |
| 1164 | `complete_batch(FAILED)` | Transform failed | trigger_type |
| 1201 | `complete_node_state(COMPLETED)` | Success | Output data |
| 1210 | `complete_batch(COMPLETED)` | Batch succeeded | trigger_type |
| 1222 | `complete_node_state(FAILED)` | Error result | Transform error |
| 1230 | `complete_batch(FAILED)` | Error result | — |

#### Recovery Methods (Read Operations)

| Line | Recorder Call | When | What's Retrieved |
|------|---------------|------|------------------|
| 1553 | `get_batch()` | Resume | Batch by batch_id |
| 1561 | `get_batch_members()` | Resume | Members for batch restore |

---

## Plugin & Client Layer

### AuditedHTTPClient (`src/elspeth/plugins/clients/http.py`)

**Automatic recording** - no plugin code required.

| Lines | When | What's Recorded |
|-------|------|-----------------|
| 308-317 | Successful POST | CallType.HTTP, status, sanitized URL, headers (fingerprinted), body, response, latency |
| 352-363 | Exception | CallType.HTTP, CallStatus.ERROR, exception details |

**Secret Handling**: Authorization headers replaced with HMAC fingerprints.

---

### AuditedLLMClient (`src/elspeth/plugins/clients/llm.py`)

**Automatic recording** - no plugin code required.

| Lines | When | What's Recorded |
|-------|------|-----------------|
| 329-337 | Successful completion | CallType.LLM, model, messages, temperature, response content, token usage |
| 384-396 | Exception | CallType.LLM, CallStatus.ERROR, error classification, retryable flag |

---

### PluginContext (`src/elspeth/plugins/context.py`)

High-level API for plugins to record audit entries.

#### `record_call()` (Lines 225-357)

Called by plugins making external calls (e.g., DatabaseSink SQL operations).

| Lines | Operation | What's Recorded |
|-------|-----------|-----------------|
| 286-301 | Node state call | state_id parent, thread-safe call_index |
| 305-315 | Operation call | operation_id parent (source/sink I/O) |

**XOR Invariant**: Exactly one of `state_id` or `operation_id` must be set.

#### `record_validation_error()` (Lines 359-431)

| When | What's Recorded |
|------|-----------------|
| Source validation failure | row_data, error message, schema mode |

#### `record_transform_error()` (Lines 433-482)

| When | What's Recorded |
|------|-----------------|
| Transform returns error | row_data, error details, token_id |

---

### DatabaseSink (`src/elspeth/plugins/sinks/database_sink.py`)

Explicitly records SQL operations.

| Lines | When | What's Recorded |
|-------|------|-----------------|
| 320-331 | Successful INSERT | CallType.SQL, table name, row count |
| 336-347 | INSERT exception | CallType.SQL, CallStatus.ERROR |

---

## Checkpoint Subsystem

### CheckpointManager (`src/elspeth/core/checkpoint/manager.py`)

| Method | Lines | When | Audit Operation |
|--------|-------|------|-----------------|
| `create_checkpoint()` | 76-107 | After sink write | INSERT checkpoints_table |
| `get_latest_checkpoint()` | 134-156 | Resume check | SELECT checkpoints_table |
| `delete_checkpoints()` | 204-206 | Run complete | DELETE checkpoints_table |

**Checkpoint Data Recorded**:
- `checkpoint_id`, `run_id`, `token_id`, `node_id`
- `sequence_number` (monotonic)
- `upstream_topology_hash` (full DAG hash)
- `checkpoint_node_config_hash`
- `aggregation_state_json` (if applicable)
- `format_version`

---

### RecoveryManager (`src/elspeth/core/checkpoint/recovery.py`)

Primarily **reads** from audit tables during resume:

| Method | Lines | What's Read |
|--------|-------|-------------|
| `can_resume()` | 83-99 | runs_table (status), checkpoints_table |
| `get_resume_point()` | 122-136 | checkpoints_table, aggregation_state |
| `get_unprocessed_rows()` | 256-365 | Complex JOIN: rows, tokens, token_outcomes |
| `get_unprocessed_row_data()` | 186-227 | rows_table, payload_store |

---

## Coalesce & Aggregation Paths

### CoalesceExecutor (`src/elspeth/engine/coalesce_executor.py`)

#### Token Arrival

| Lines | When | Recorder Calls |
|-------|------|----------------|
| 257-264 | Token arrives (held) | `begin_node_state()` |
| 205-217 | Late arrival (after merge) | `begin_node_state()`, `complete_node_state(FAILED)`, `record_token_outcome(FAILED)` |

#### Merge Execution

| Lines | When | Recorder Calls |
|-------|------|----------------|
| 399-413 | Merge success | `complete_node_state(COMPLETED)`, `record_token_outcome(COALESCED)` |
| 328-343 | Select branch missing | `complete_node_state(FAILED)`, `record_token_outcome(FAILED)` |

#### Timeout & End-of-Source

| Lines | When | Recorder Calls |
|-------|------|----------------|
| 551-562 | Timeout, quorum not met | `complete_node_state(FAILED)`, `record_token_outcome(FAILED)` |
| 658-671 | EOS, quorum not met | `complete_node_state(FAILED)`, `record_token_outcome(FAILED)` |
| 709-722 | EOS, require_all incomplete | `complete_node_state(FAILED)`, `record_token_outcome(FAILED)` |

---

### TokenManager (`src/elspeth/engine/tokens.py`)

Atomic lineage operations:

| Method | Recorder Call | What's Recorded |
|--------|---------------|-----------------|
| `fork_token()` | `recorder.fork_token()` | Child tokens, parent relationships, FORKED outcome |
| `coalesce_tokens()` | `recorder.coalesce_tokens()` | Merged token, parent relationships, join_group_id |
| `expand_token()` | `recorder.expand_token()` | Child tokens, parent relationships, optional EXPANDED outcome |

---

## LandscapeRecorder API Reference

### Recording Methods by Category

#### Run Lifecycle (6 methods)
- `begin_run()` → `runs` INSERT
- `complete_run()` → `runs` UPDATE
- `update_run_status()` → `runs` UPDATE
- `set_export_status()` → `runs` UPDATE
- `record_source_field_resolution()` → `runs` UPDATE
- `finalize_run()` → `runs` UPDATE + grade computation

#### Node & Edge Registration (2 methods)
- `register_node()` → `nodes` INSERT
- `register_edge()` → `edges` INSERT

#### Row & Token Creation (5 methods)
- `create_row()` → `rows` INSERT + payload persist
- `create_token()` → `tokens` INSERT
- `fork_token()` → `tokens` + `token_parents` + `token_outcomes` INSERT (atomic)
- `coalesce_tokens()` → `tokens` + `token_parents` INSERT (atomic)
- `expand_token()` → `tokens` + `token_parents` + optional `token_outcomes` INSERT (atomic)

#### Node State (2 methods)
- `begin_node_state()` → `node_states` INSERT
- `complete_node_state()` → `node_states` UPDATE

#### Routing (2 methods)
- `record_routing_event()` → `routing_events` INSERT
- `record_routing_events()` → `routing_events` INSERT (multiple, atomic)

#### Token Outcomes (1 method)
- `record_token_outcome()` → `token_outcomes` INSERT

#### Batch/Aggregation (5 methods)
- `create_batch()` → `batches` INSERT
- `add_batch_member()` → `batch_members` INSERT
- `update_batch_status()` → `batches` UPDATE
- `complete_batch()` → `batches` UPDATE
- `retry_batch()` → `batches` UPDATE + `batch_members` for new attempt

#### External Calls (2 methods)
- `record_call()` → `calls` INSERT (state_id parent)
- `record_operation_call()` → `calls` INSERT (operation_id parent)

#### Operations (2 methods)
- `begin_operation()` → `operations` INSERT
- `complete_operation()` → `operations` UPDATE

#### Errors (2 methods)
- `record_validation_error()` → `validation_errors` INSERT
- `record_transform_error()` → `transform_errors` INSERT

#### Artifacts (1 method)
- `register_artifact()` → `artifacts` INSERT

---

## Timing Diagrams

### Normal Row Processing

```
Source yields row
    │
    ├─► Orchestrator: create_row(), create_token()
    │
    ▼
Transform 1 processes
    │
    ├─► TransformExecutor: begin_node_state()
    │   [transform.process() executes]
    │   [AuditedLLMClient: record_call() - if LLM used]
    ├─► TransformExecutor: complete_node_state(COMPLETED)
    │
    ▼
Transform N processes
    │
    ├─► (same pattern)
    │
    ▼
Sink writes
    │
    ├─► SinkExecutor: begin_node_state()
    │   [sink.write() executes]
    ├─► SinkExecutor: complete_node_state(COMPLETED)
    ├─► SinkExecutor: register_artifact()
    ├─► SinkExecutor: record_token_outcome(COMPLETED)
    │
    ├─► CheckpointManager: create_checkpoint() [if enabled]
    │
    ▼
Row complete
```

### Fork/Join Processing

```
Token reaches gate with fork
    │
    ├─► GateExecutor: begin_node_state()
    ├─► GateExecutor: complete_node_state(COMPLETED)
    ├─► GateExecutor: record_routing_events() [multiple routes]
    │
    ├─► TokenManager: fork_token() [ATOMIC]
    │       ├─► Creates child tokens
    │       ├─► Records parent relationships
    │       └─► Records FORKED outcome for parent
    │
    ▼
Each child token processes independently
    │
    ▼
Children arrive at coalesce point
    │
    ├─► CoalesceExecutor: begin_node_state() [for each arrival]
    │   [Waits for all/quorum branches]
    │
    ▼
Merge triggers
    │
    ├─► TokenManager: coalesce_tokens() [ATOMIC]
    │       ├─► Creates merged token
    │       └─► Records parent relationships
    │
    ├─► CoalesceExecutor: complete_node_state(COMPLETED) [for each consumed]
    ├─► CoalesceExecutor: record_token_outcome(COALESCED) [for each consumed]
    │
    ▼
Merged token continues processing
```

### Aggregation Processing

```
First row arrives at aggregation
    │
    ├─► AggregationExecutor: create_batch()
    ├─► AggregationExecutor: add_batch_member()
    ├─► Processor: record_token_outcome(BUFFERED) [non-terminal]
    │
    ▼
Additional rows arrive
    │
    ├─► AggregationExecutor: add_batch_member() [for each]
    ├─► Processor: record_token_outcome(BUFFERED/CONSUMED_IN_BATCH)
    │
    ▼
Trigger fires (count/timeout/EOS)
    │
    ├─► AggregationExecutor: update_batch_status(EXECUTING)
    ├─► AggregationExecutor: begin_node_state()
    │   [transform.process() with batch_rows]
    ├─► AggregationExecutor: complete_node_state(COMPLETED)
    ├─► AggregationExecutor: complete_batch(COMPLETED)
    │
    ▼
Result tokens continue processing
```

---

## Key Invariants

### 1. Every Token Reaches Terminal State

No silent drops. Every token eventually records one of:
- `COMPLETED` - Reached output sink
- `ROUTED` - Sent to named sink
- `FORKED` - Split (parent delegates to children)
- `EXPANDED` - Deaggregated (parent delegates to children)
- `CONSUMED_IN_BATCH` - Aggregated
- `COALESCED` - Merged in join
- `QUARANTINED` - Failed validation, stored
- `FAILED` - Unrecoverable error

### 2. XOR Call Attribution

Every external call has exactly ONE parent:
- `state_id` (transform processing) **OR**
- `operation_id` (source/sink I/O)

Never both. Never neither.

### 3. Atomic Lineage Operations

Fork, coalesce, and expand are transactional:
- All child tokens created together
- All parent relationships recorded together
- Parent outcome recorded in same transaction

### 4. Telemetry After Audit

Telemetry emission always occurs AFTER successful audit recording:
- Wrapped in try/except
- Telemetry failures never corrupt audit trail
- Audit is the source of truth

---

*Last updated: 2026-01-31*
