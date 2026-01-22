# Subsystem Catalog

**Analysis Date:** 2026-01-21
**Analyst:** Claude Code (Opus 4.5)
**Total Source LOC:** ~25,000

This catalog provides detailed architectural documentation for each ELSPETH subsystem, synthesized from parallel deep-dive analysis.

---

## Table of Contents

1. [Contracts](#1-contracts)
2. [Core Utilities](#2-core-utilities)
3. [Landscape (Audit Trail)](#3-landscape-audit-trail)
4. [Engine](#4-engine)
5. [Plugin System](#5-plugin-system)
6. [Plugin Implementations](#6-plugin-implementations)
7. [Production Operations](#7-production-operations)
8. [CLI/TUI](#8-clitui)

---

## 1. Contracts

**Location:** `src/elspeth/contracts/`
**LOC:** ~2,000
**Independence:** HIGH - pure data models

### Responsibility

Defines all shared data structures, enums, and type contracts that cross subsystem boundaries, serving as the single source of truth for inter-module type definitions.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 173 | Central re-export hub; 60+ types grouped by category |
| `enums.py` | 219 | Status codes and classification enums |
| `audit.py` | 419 | Landscape database record contracts |
| `results.py` | 324 | Operation outcome types with factory methods |
| `data.py` | 243 | Pydantic-based plugin schema validation |
| `routing.py` | 148 | Flow control contracts (RoutingAction, EdgeInfo) |
| `errors.py` | 42 | TypedDict schemas for structured error payloads |

### Public API

**Enums:**
- `RunStatus`, `NodeStateStatus`, `BatchStatus`, `ExportStatus`
- `NodeType`, `Determinism`, `RoutingKind`, `RoutingMode`
- `RowOutcome`, `CallType`, `CallStatus`, `RunMode`, `TriggerType`

**Audit Records:**
- `Run`, `Node`, `Edge`, `Row`, `Token`, `TokenParent`
- `NodeState` (discriminated union: `NodeStateOpen | NodeStateCompleted | NodeStateFailed`)
- `Call`, `Artifact`, `Batch`, `BatchMember`, `BatchOutput`
- `Checkpoint`, `RoutingEvent`, `TokenOutcome`
- `ValidationErrorRecord`, `TransformErrorRecord`, `RowLineage`

**Results:**
- `TransformResult` (factory: `.success()`, `.error()`, `.success_multi()`)
- `GateResult`, `RowResult`, `SourceRow`, `ArtifactDescriptor`, `FailureInfo`

**Routing:**
- `RoutingAction` (factory: `.continue_()`, `.route()`, `.fork_to_paths()`)
- `RoutingSpec`, `EdgeInfo`

### Internal Patterns

1. **Discriminated Unions:** `NodeState = NodeStateOpen | NodeStateCompleted | NodeStateFailed` with `status` field as discriminator
2. **Factory Methods:** Constrained construction prevents invalid states
3. **Frozen Dataclasses:** Immutable audit records enforce integrity
4. **TypedDict for Updates:** Partial update schemas (`ExportStatusUpdate`, `BatchStatusUpdate`)
5. **`(str, Enum)` Base:** All enums enable direct database serialization via `.value`

### Dependencies

- **Inbound:** 59 source files + 96 test files depend on contracts
- **Outbound:** Only `elspeth.core.config` (re-export), `pydantic`, standard library

### Confidence: HIGH

Read 100% of files (12 files, 1951 lines). Verified via import analysis and cross-reference with CLAUDE.md trust model.

---

## 2. Core Utilities

**Location:** `src/elspeth/core/` (top-level files)
**LOC:** ~2,000
**Independence:** MEDIUM - foundational

### Responsibility

Provides foundational infrastructure for deterministic hashing, configuration management, DAG validation, structured logging, and content-addressable blob storage.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `canonical.py` | 148 | Two-phase RFC 8785 JSON canonicalization |
| `config.py` | 1186 | Dynaconf + Pydantic configuration hierarchy |
| `dag.py` | 579 | NetworkX-based DAG validation and traversal |
| `logging.py` | 79 | structlog wrapper for structured logging |
| `payload_store.py` | 156 | Content-addressable blob storage protocol |

### Canonical JSON (Two-Phase)

```
Phase 1: Normalize (ours)       Phase 2: Serialize (rfc8785)
├─ pandas/numpy → primitives    └─ RFC 8785/JCS standard
├─ datetime → UTC ISO string        ├─ Sorted keys
├─ bytes → base64                   ├─ No whitespace
└─ NaN/Infinity → REJECT            └─ Deterministic output
```

**Key Functions:**
- `canonical_json(obj) -> str` - Deterministic JSON
- `stable_hash(obj) -> str` - SHA-256 hex digest

### Configuration Hierarchy

```python
ElspethSettings (frozen)
├── DatasourceSettings      # Source plugin config
├── SinkSettings            # Output plugins (dict by name)
├── RowPluginSettings       # Transform chain (ordered list)
├── GateSettings            # Config-driven routing
├── AggregationSettings     # Batching with triggers
├── CoalesceSettings        # Fork path merging
├── LandscapeSettings       # Audit backend selection
├── CheckpointSettings      # Crash recovery
├── RetrySettings           # Exponential backoff
└── RateLimitSettings       # External call throttling
```

**Precedence (high → low):**
1. Environment variables (`ELSPETH_*`)
2. Config file (settings.yaml)
3. Pydantic defaults

### DAG Validation

- **NetworkX MultiDiGraph** for multiple edge support (fork gates)
- **Validation rules:** Acyclicity, single source, at least one sink, unique labels per node
- **ID maps:** `_sink_id_map`, `_transform_id_map`, `_config_gate_id_map`, `_aggregation_id_map`, `_coalesce_id_map`
- **Route resolution:** `_route_resolution_map[(gate_node, label) -> target]`

### Payload Store

- **Content-addressable:** Stored by SHA-256 hash for deduplication
- **Protocol + Implementation:** `PayloadStore` protocol + `FilesystemPayloadStore`
- **Integrity verification:** `hmac.compare_digest()` for timing-safe comparison
- **Structure:** `base_path/{hash[0:2]}/{hash}` for file distribution

### Dependencies

- **Inbound:** CLI, Engine, Plugins, Landscape, Checkpoint, Contracts
- **Outbound:** `rfc8785`, `networkx`, `structlog`, `pydantic`, `dynaconf`

### Patterns

- **Defense-in-depth:** NaN rejection, timing-safe hash comparison
- **Versioned canonicalization:** `CANONICAL_VERSION` enables future migration
- **Expression validation:** Gate conditions validated at config time
- **Secret fingerprinting:** Separates audit-safe from runtime config

### Confidence: HIGH

Read 100% of all 5 utility files. Cross-verified dependencies via grep.

---

## 3. Landscape (Audit Trail)

**Location:** `src/elspeth/core/landscape/`
**LOC:** ~4,600
**Independence:** HIGH - self-contained

### Responsibility

Provides the audit backbone for ELSPETH's SDA pipelines, recording every operation with complete traceability to ensure decisions can be explained and verified for formal inquiry.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `recorder.py` | 2,571 | Primary high-level audit API (60+ methods) |
| `schema.py` | 359 | SQLAlchemy Core table definitions (16 tables) |
| `exporter.py` | 382 | Compliance/legal export with HMAC signing |
| `models.py` | 334 | Dataclass model documentation |
| `repositories.py` | 234 | DB-to-domain conversion with enum coercion |
| `database.py` | 210 | Connection management (SQLite/PostgreSQL) |
| `lineage.py` | 149 | `explain()` function for complete lineage |
| `reproducibility.py` | 136 | Grade computation (FULL/REPLAY/ATTRIBUTABLE) |

### Database Schema (16 Tables)

| Table | Purpose | Key Relationships |
|-------|---------|-------------------|
| `runs` | Pipeline execution metadata | Root entity |
| `nodes` | Plugin instances in graph | FK to runs |
| `edges` | Connections between nodes | FK to runs, nodes |
| `rows` | Source rows loaded | FK to runs, nodes |
| `tokens` | Row instances in DAG paths | FK to rows |
| `token_outcomes` | Terminal state recording (AUD-001) | FK to runs, tokens |
| `token_parents` | Multi-parent relationships | FK to tokens (both sides) |
| `node_states` | Processing records at each node | FK to tokens, nodes |
| `calls` | External API calls | FK to node_states |
| `artifacts` | Sink outputs | FK to runs, node_states |
| `routing_events` | Gate routing decisions | FK to node_states, edges |
| `batches` | Aggregation batch tracking | FK to runs, nodes |
| `batch_members` | Tokens in each batch | FK to batches, tokens |
| `batch_outputs` | Outputs from batches | FK to batches |
| `validation_errors` | Source validation failures | FK to runs |
| `transform_errors` | Transform processing errors | FK to runs |
| `checkpoints` | Crash recovery checkpoints | FK to runs, tokens |

### LandscapeRecorder API

**Recording Methods:**
- `begin_run()`, `complete_run()`, `fail_run()`
- `register_node()`, `register_edge()`
- `create_row()`, `create_token()`, `fork_token()`, `coalesce_tokens()`, `expand_token()`
- `begin_node_state()`, `complete_node_state()`
- `record_routing_event()`, `record_call()`, `record_token_outcome()`
- `create_batch()`, `add_batch_member()`, `complete_batch()`
- `register_artifact()`
- `record_validation_error()`, `record_transform_error()`

**Query Methods:**
- `get_run()`, `get_nodes()`, `get_rows()`, `get_tokens()`
- `get_node_states_for_token()`, `get_routing_events()`, `get_calls()`
- `explain_row()`, `find_call_by_request_hash()`, `get_call_response_data()`

### Three-Tier Trust Enforcement

```
Tier 1 (Landscape): CRASH on any anomaly
├─ _row_to_node_state() raises ValueError on COMPLETED with NULL output_hash
├─ _coerce_enum() fails fast on invalid values
└─ No defensive handling in repository layer

Tier 3 (Purged data): Graceful degradation
├─ RowLineage.payload_available = False when purged
└─ RowDataResult provides explicit state (AVAILABLE/PURGED/NEVER_STORED)
```

### Dependencies

- **Inbound:** 28 files depend on Landscape (engine, plugins, checkpoint, CLI, TUI)
- **Outbound:** `elspeth.contracts`, `elspeth.core.canonical`, `sqlalchemy`

### Patterns

- **SQLAlchemy Core (not ORM):** Explicit SQL control, no magic
- **Event-driven recording:** begin/complete pattern with timing
- **Repository pattern:** Separate DB-to-domain conversion layer
- **Signed exports:** HMAC-SHA256 with hash chains for tamper detection
- **Terminal outcome enforcement:** Partial unique index ensures one terminal per token

### Concerns

1. **Repository layer incomplete:** Repositories defined but not consistently used
2. **Large file:** `recorder.py` at 2,571 LOC could be split by entity type
3. **SQLite-only validation:** Schema validation only checks SQLite databases

### Confidence: HIGH

Read 100% of all 11 Python files (4,618 total LOC).

---

## 4. Engine

**Location:** `src/elspeth/engine/`
**LOC:** ~5,900
**Independence:** MEDIUM - depends on contracts, landscape

### Responsibility

Orchestrates the full SDA pipeline run lifecycle, managing token flow through transforms/gates/aggregations, recording audit trails, handling fork/join/coalesce operations, and coordinating sink writes with checkpoint support.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `orchestrator.py` | 1,622 | Full run lifecycle management |
| `executors.py` | 1,340 | Plugin execution wrappers with audit |
| `processor.py` | 1,014 | Row-level processing via work queue |
| `expression_parser.py` | 464 | Safe AST-based gate condition parsing |
| `coalesce_executor.py` | 453 | Stateful barrier for merging forks |
| `tokens.py` | 254 | TokenManager for token lifecycle |
| `spans.py` | 240 | OpenTelemetry span factory |
| `retry.py` | 182 | RetryManager using tenacity |
| `triggers.py` | 153 | Aggregation trigger evaluation |
| `schema_validator.py` | 102 | Pipeline schema compatibility |

### Execution Flow

```
1. Run Initialization
   └── Orchestrator.run() creates LandscapeRecorder, begins run

2. Graph Registration
   └── Register nodes/edges with Landscape, build edge_map

3. Validation
   └── Validate routes, error sinks, quarantine destinations

4. Plugin Lifecycle
   └── Call on_start() on source, transforms, sinks

5. Row Processing Loop
   └── For each source row:
       ├── Create token via TokenManager
       ├── Process via RowProcessor (work queue for DAG)
       └── Handle fork children, aggregation buffering

6. End-of-Source Flush
   └── Flush aggregation buffers, coalesce operations

7. Sink Writes
   └── SinkExecutor writes pending tokens, checkpoints

8. Run Completion
   └── on_complete(), close plugins, delete checkpoints
```

### Token State Machine

```
┌─────────────────────────────────────────────────────────┐
│                   Token Outcomes                         │
├─────────────────────────────────────────────────────────┤
│ COMPLETED       - Reached output sink                   │
│ ROUTED          - Gate routed to named sink             │
│ FORKED          - Split to multiple paths               │
│ CONSUMED_IN_BATCH - Aggregated into batch               │
│ COALESCED       - Merged from parallel paths            │
│ QUARANTINED     - Failed validation                     │
│ FAILED          - Unrecoverable failure                 │
│ EXPANDED        - Deaggregation parent (1->N)           │
│ BUFFERED        - Passthrough waiting for batch         │
└─────────────────────────────────────────────────────────┘
```

### Work Queue Pattern

```python
# processor.py: DAG traversal via work queue
work_queue: deque[_WorkItem] = deque()
work_queue.append(_WorkItem(token, start_step=0))

while work_queue:
    item = work_queue.popleft()
    result, children = _process_single_token(item)
    work_queue.extend(children)  # Fork children queued
```

### Dependencies

- **Inbound:** `elspeth.cli`, `elspeth.core.checkpoint`
- **Outbound:** `elspeth.contracts`, `elspeth.core.landscape`, `elspeth.core.dag`, `elspeth.core.config`, `elspeth.plugins.*`

### Patterns

- **Stateless recovery:** `resume()` creates fresh recorder/processor
- **Validation-first:** All destinations validated before processing
- **Audit-first:** Every operation recorded before/after
- **Defensive token isolation:** `copy.deepcopy()` on fork
- **Control flow signals:** `BatchPendingError` for async batch coordination

### Concerns

1. **Resume code duplication:** `_process_resumed_rows()` duplicates `_execute_run()` logic
2. **Large file:** `orchestrator.py` at 1,622 LOC could be split
3. **Magic constants:** `MAX_WORK_QUEUE_ITERATIONS = 10_000` should be configurable

### Confidence: HIGH

Read all 12 files (~5,900 lines). Traced execution flow through orchestrator → processor → executors.

---

## 5. Plugin System

**Location:** `src/elspeth/plugins/` (infrastructure files)
**LOC:** ~2,300
**Independence:** HIGH - interface definitions

### Responsibility

Provides the pluggy-based plugin infrastructure defining contracts (protocols), base implementations, discovery mechanisms, typed configuration, and runtime context.

### Key Components

| File | LOC | Purpose |
|------|-----|---------|
| `protocols.py` | 456 | `@runtime_checkable` Protocol classes |
| `context.py` | 375 | `PluginContext` execution context |
| `base.py` | 329 | ABC base classes with defaults |
| `discovery.py` | 272 | Filesystem-based plugin discovery |
| `manager.py` | 242 | pluggy wrapper and registration |
| `config_base.py` | 168 | Pydantic config hierarchy |
| `schema_factory.py` | 151 | Runtime Pydantic schema creation |
| `hookspecs.py` | 82 | pluggy hook specifications |

### Plugin Protocols

| Protocol | Method | Schemas | Purpose |
|----------|--------|---------|---------|
| `SourceProtocol` | `load(ctx)` | `output_schema` | Load external data |
| `TransformProtocol` | `process(row, ctx)` | in/out | Stateless row processing |
| `GateProtocol` | `evaluate(row, ctx)` | in/out | Routing decisions |
| `CoalesceProtocol` | `merge(outputs, ctx)` | `output_schema` | Merge parallel paths |
| `SinkProtocol` | `write(rows, ctx)` | `input_schema` | Output data |

### Discovery Mechanism

```
discover_all_plugins()
├── Scan PLUGIN_SCAN_CONFIG directories
├── Load modules via importlib.util
├── Find classes inheriting from Base* with name attribute
├── Duplicate names crash immediately (system-owned code)
└── Gates excluded (config-driven engine operations)
```

### PluginContext

```python
@dataclass
class PluginContext:
    run_id: str
    config: ElspethSettings

    # Phase 3 integrations
    landscape: LandscapeRecorder | None
    tracer: SpanFactory | None
    payload_store: PayloadStore | None

    # Phase 6 additions
    state_id: str | None
    llm_client: AuditedLLMClient | None
    http_client: AuditedHTTPClient | None

    # Methods
    get(key: str) -> Any          # Dotted config access
    start_span(name) -> Span      # OpenTelemetry
    record_call(...)              # Audit trail
    get_checkpoint() -> dict      # Batch transform state
    update_checkpoint(state)      # Persist state
```

### Schema Factory

```python
create_schema_from_config(schema_config, name, allow_coercion)
├── Dynamic mode: extra="allow", no type validation
├── Strict mode: extra="forbid", explicit fields only
├── Free mode: extra="allow", requires specified fields
└── allow_coercion: Sources coerce (True), Transforms reject (False)
```

### Dependencies

- **Inbound:** Engine (orchestrator, processor, executors), all plugin implementations
- **Outbound:** `elspeth.contracts`, `elspeth.core.canonical`, `pluggy`

### Patterns

- **Protocol + ABC:** Protocols for type checking, ABCs for convenience implementations
- **Three-tier enforcement:** `allow_coercion` parameter enforces trust model
- **Structural aggregation:** `is_batch_aware=True` flag, not separate class
- **Frozen metadata:** `PluginSpec` immutable after registration
- **Deferred imports:** TYPE_CHECKING guards avoid circular imports

### Confidence: HIGH

Read 100% of infrastructure files (12 files, 2,326 LOC).

---

## 6. Plugin Implementations

**Location:** `src/elspeth/plugins/{sources,transforms,sinks,llm,azure,clients,pooling}/`
**LOC:** ~7,000
**Independence:** HIGH - isolated implementations

### Responsibility

Provides concrete implementations of data sources, transforms, and sinks, with specialized packs for LLM integration, Azure services, and shared infrastructure.

### Source Plugins

| Plugin | LOC | Description |
|--------|-----|-------------|
| `csv_source.py` | 136 | pandas-based CSV loading with validation |
| `json_source.py` | 197 | JSON array/JSONL with auto-detection |
| `null_source.py` | 69 | Empty source for resume operations |

### Transform Plugins

| Plugin | LOC | Description |
|--------|-----|-------------|
| `passthrough.py` | 92 | Identity transform for testing |
| `field_mapper.py` | 128 | Field renaming/selection |
| `keyword_filter.py` | 172 | Regex content filtering |
| `batch_stats.py` | 157 | Batch aggregation (count/sum/mean) |
| `batch_replicate.py` | 145 | Batch deaggregation |
| `json_explode.py` | 163 | Array explosion (1→N) |
| `azure/content_safety.py` | 618 | Azure Content Safety API |
| `azure/prompt_shield.py` | 556 | Jailbreak detection |

### Sink Plugins

| Plugin | LOC | Description |
|--------|-----|-------------|
| `csv_sink.py` | 245 | CSV with SHA-256 content hash |
| `json_sink.py` | 188 | JSON/JSONL with hash |
| `database_sink.py` | 255 | SQLAlchemy Core dynamic tables |

### LLM Pack

| Plugin | LOC | Description |
|--------|-----|-------------|
| `base.py` | 277 | Abstract `BaseLLMTransform` with Jinja2 |
| `azure.py` | 597 | Azure OpenAI with pooling |
| `openrouter.py` | 669 | Multi-model routing (100+ models) |
| `azure_batch.py` | 723 | Azure Batch API (50% savings) |
| `azure_multi_query.py` | 520 | Cross-product evaluation |
| `templates.py` | 189 | Jinja2 with audit hashing |

### Azure Pack

| Plugin | LOC | Description |
|--------|-----|-------------|
| `auth.py` | 215 | Four auth methods (SAS, managed identity, etc.) |
| `blob_source.py` | 483 | Azure Blob Storage source |
| `blob_sink.py` | 448 | Azure Blob Storage sink |

### Client Infrastructure

| Component | LOC | Description |
|-----------|-----|-------------|
| `AuditedLLMClient` | 226 | OpenAI wrapper with call recording |
| `AuditedHTTPClient` | 222 | httpx wrapper with header filtering |
| `CallReplayer` | 231 | Replay mode from recorded calls |
| `CallVerifier` | 224 | Verify mode with DeepDiff |

### Pooling Infrastructure

| Component | LOC | Description |
|-----------|-----|-------------|
| `PooledExecutor` | 317 | ThreadPool with semaphore control |
| `AIMDThrottle` | 156 | TCP-style congestion control |
| `ReorderBuffer` | — | Strict output ordering |

### Patterns

- **Three-tier trust:** Sources use `allow_coercion=True`, transforms reject
- **Self-contained clients:** LLM transforms create own `AuditedLLMClient`
- **Pooled execution:** `pool_size>1` via shared `PooledExecutor`
- **Batch-aware transforms:** `is_batch_aware=True`, `creates_tokens=True`
- **Content hashing:** All sinks compute SHA-256 for artifacts

### Concerns

1. **Azure pooling duplication:** ~500 lines duplicated between Content Safety and Prompt Shield
2. **LLM batch logic duplication:** Similar patterns across Azure/OpenRouter transforms
3. **Auth field duplication:** Blob source/sink repeat auth fields instead of composing

### Confidence: HIGH

Read 100% of plugin implementations across all subdirectories.

---

## 7. Production Operations

**Location:** `src/elspeth/core/{checkpoint,retention,rate_limit,security}/`
**LOC:** ~1,200
**Independence:** HIGH - independent modules

### Checkpoint/Recovery

**Files:** `manager.py` (155), `recovery.py` (273), `__init__.py` (13)

**How It Works:**
1. During execution, `CheckpointManager.create_checkpoint()` records `(run_id, token_id, node_id, sequence_number)` with optional aggregation state JSON
2. `RecoveryManager.can_resume()` validates: run exists, status=FAILED, checkpoints exist
3. `get_resume_point()` loads latest checkpoint, deserializes aggregation state
4. `get_unprocessed_rows()` finds rows after checkpoint boundary
5. After success, `delete_checkpoints()` cleans up

### Retention/Purge

**Files:** `purge.py` (247), `__init__.py` (10)

**Purge Strategy:**
1. Compute cutoff: `as_of - timedelta(days=retention_days)`
2. Query expired payload refs from 4 sources via UNION:
   - `rows.source_data_ref`
   - `calls.request_ref`
   - `calls.response_ref`
   - `routing_events.reason_ref`
3. Filter: `runs.status == "completed"` AND `runs.completed_at < cutoff`
4. Delete blobs; hashes remain in Landscape for integrity verification

### Rate Limiting

**Files:** `limiter.py` (297), `registry.py` (122), `__init__.py` (9)

**Features:**
- Dual rate support (per-second and per-minute)
- `InMemoryBucket` or `SQLiteBucket` for cross-process coordination
- `RateLimitRegistry` lazy creates limiters per service name
- `NoOpLimiter` when disabled

### Security (Fingerprinting)

**Files:** `fingerprint.py` (135), `__init__.py` (6)

**Key Resolution:**
1. Environment variable: `ELSPETH_FINGERPRINT_KEY`
2. Azure Key Vault: `ELSPETH_KEYVAULT_URL` + `ELSPETH_KEYVAULT_SECRET_NAME`

**Usage:** `secret_fingerprint(secret, key)` → 64-char HMAC-SHA256 hex

### Dependencies

- **Inbound:** Engine (Orchestrator.resume), CLI
- **Outbound:** `elspeth.contracts`, `elspeth.core.landscape`, `elspeth.core.payload_store`, `pyrate-limiter`, `azure-keyvault-secrets` (optional)

### Patterns

- **Protocol for dependency inversion:** `PayloadStoreProtocol` in purge
- **Frozen dataclasses with validation:** `ResumeCheck.__post_init__` invariant
- **Context manager support:** Limiters implement `__enter__/__exit__`
- **Custom excepthook:** Rate limiter suppresses pyrate-limiter cleanup race

### Concerns

1. **PurgeResult.bytes_freed always 0:** Protocol doesn't return size
2. **No checkpoint interval config:** Caller discretion only
3. **Rate limit cleanup race:** Requires custom excepthook workaround

### Confidence: HIGH

Read all 10 Python files across all 4 subsystems.

---

## 8. CLI/TUI

**Location:** `src/elspeth/cli.py`, `src/elspeth/tui/`
**LOC:** ~2,000
**Independence:** MEDIUM - integrates all subsystems

### CLI

**File:** `cli.py` (1,140 LOC)

**Commands:**

| Command | Description |
|---------|-------------|
| `run` | Execute pipeline (requires `--execute` flag) |
| `explain` | Launch TUI or output lineage as JSON/text |
| `validate` | Validate configuration without running |
| `plugins list` | List available plugins |
| `purge` | Delete old payloads per retention policy |
| `resume` | Resume failed run from checkpoint |
| `health` | System health check |

**Configuration Loading:**
- `load_settings()` via Dynaconf + Pydantic
- Supports `--settings`/`-s`, `.env` loading, tilde expansion

### TUI

**Location:** `src/elspeth/tui/` (~650 LOC across 9 files)

**Components:**

| Component | Purpose |
|-----------|---------|
| `ExplainApp` | Main Textual App with two-column layout |
| `ExplainScreen` | Lineage visualization with state machine |
| `LineageTree` | Tree widget (Run > Source > Transforms > Sinks > Tokens) |
| `NodeDetailPanel` | Detail panel (identity, status, hashes, errors, artifacts) |

**State Machine Pattern:**
```python
# ExplainScreen uses discriminated union for state
ExplainState = UninitializedState | LoadingFailedState | LoadedState
```

### Dependencies

- **Inbound:** Entry point for user interaction
- **Outbound:** All subsystems (config, dag, landscape, engine, plugins, checkpoint, retention)

### Patterns

- **Safety flag:** `run` requires explicit `--execute`
- **Singleton pattern:** Plugin manager cached at module level
- **Discriminated union state:** Type-safe screen state management
- **TypedDict contracts:** TUI data types enforce direct field access

### Concerns

1. **TUI partially implemented:** Widgets exist but not wired into main app
2. **`explain` incomplete:** JSON/text output exits with code 2

### Confidence: HIGH

Read 100% of CLI (1,140 lines) and TUI (9 files, ~650 lines).

---

## Cross-Subsystem Dependency Matrix

```
                    Contracts  Core  Landscape  Engine  Plugins  Prod-Ops  CLI
Contracts              -       ✓        ✓         ✓        ✓        ✓       ✓
Core Utilities         ✓       -        ✓         ✓        ✓        ✓       ✓
Landscape              ✓       ✓        -         ✓        ✓        ✓       ✓
Engine                 ✓       ✓        ✓         -        ✓        ✓       ✓
Plugin System          ✓       ✓        -         ✓        -        -       ✓
Plugin Impls           ✓       ✓        ✓         -        ✓        -       -
Production Ops         ✓       ✓        ✓         ✓        -        -       ✓
CLI/TUI                ✓       ✓        ✓         ✓        ✓        ✓       -
```

**Legend:** ✓ = depends on (reads from)

---

## Summary Statistics

| Subsystem | Files | LOC | Independence |
|-----------|-------|-----|--------------|
| Contracts | 12 | ~2,000 | HIGH |
| Core Utilities | 5 | ~2,000 | MEDIUM |
| Landscape | 11 | ~4,600 | HIGH |
| Engine | 12 | ~5,900 | MEDIUM |
| Plugin System | 12 | ~2,300 | HIGH |
| Plugin Implementations | ~45 | ~7,000 | HIGH |
| Production Ops | 10 | ~1,200 | HIGH |
| CLI/TUI | 10 | ~2,000 | MEDIUM |
| **Total** | **~117** | **~25,000** | — |
