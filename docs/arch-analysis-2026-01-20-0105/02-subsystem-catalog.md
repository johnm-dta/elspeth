# Subsystem Catalog: ELSPETH Architecture

This document provides detailed documentation for each major subsystem in the ELSPETH framework.

---

## 1. CLI Subsystem

**Location:** `src/elspeth/cli.py`

**Responsibility:** Provide command-line interface for pipeline execution, validation, and audit operations.

**Key Components:**

| File | Description |
|------|-------------|
| `cli.py` | Typer-based CLI with commands: run, explain, validate, purge, resume, plugins |

**Public API:**

```python
# Commands
elspeth run --settings settings.yaml --execute    # Execute pipeline
elspeth explain --run <id> --token <token>        # Query lineage
elspeth validate --settings settings.yaml          # Validate config
elspeth purge --retention-days 90                  # Delete old payloads
elspeth resume <run_id>                            # Resume failed run
elspeth plugins list                               # List available plugins
```

**Dependencies:**

- **Inbound:** None (entry point)
- **Outbound:** Core (config, dag), Engine (orchestrator), Plugins, TUI, Landscape

**Patterns Observed:**

- Safety-first design: `--execute` flag required to run pipelines
- Dry-run mode for validation without execution
- Plugin registry is hardcoded (Phase 4 limitation)
- Graceful error handling with exit codes

**Concerns:**

- Plugin discovery is static (TRANSFORM_PLUGINS dict hardcoded)
- Resume implementation is partial (validation only, no actual resumption)

**Confidence:** High - Well-structured Typer application with clear command patterns

---

## 2. Contracts Subsystem

**Location:** `src/elspeth/contracts/`

**Responsibility:** Define cross-boundary data types for audit records, results, routing, and configuration. All dataclasses, enums, and TypedDicts that cross subsystem boundaries are defined here.

**Key Components:**

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports all contracts (single import point) |
| `audit.py` | Run, Node, Edge, Token, Row, NodeState, Batch, Artifact |
| `enums.py` | RunStatus, RowOutcome, NodeType, Determinism, BatchStatus, etc. |
| `results.py` | TransformResult, GateResult, SourceRow, ArtifactDescriptor, FailureInfo |
| `routing.py` | RoutingAction, RoutingSpec, EdgeInfo |
| `config.py` | Settings dataclasses (ElspethSettings, LandscapeSettings, etc.) |
| `errors.py` | ExecutionError, RoutingReason, TransformReason |
| `identity.py` | TokenInfo (row_id, token_id, row_data, branch_name) |
| `data.py` | PluginSchema, validate_row, check_compatibility |
| `schema.py` | FieldDefinition, SchemaConfig |
| `engine.py` | RetryPolicy |
| `cli.py` | ExecutionResult TypedDict |

**Public API:**

```python
from elspeth.contracts import (
    # Audit records
    Run, Node, Edge, Token, Row, NodeState, Batch, Artifact,
    # Results
    TransformResult, GateResult, SourceRow, RowResult, TokenInfo,
    # Enums
    RunStatus, RowOutcome, NodeType, Determinism,
    # Routing
    RoutingAction, RoutingSpec, EdgeInfo,
    # Schemas
    PluginSchema, validate_row,
)
```

**Dependencies:**

- **Inbound:** All other subsystems
- **Outbound:** None (leaf node - no internal dependencies)

**Patterns Observed:**

- Discriminated union types (NodeState → Open | Completed | Failed)
- Dataclasses with frozen=True where immutability matters
- Enum subclassing str for JSON serialization compatibility
- Factory methods on result types (`TransformResult.success()`, `TransformResult.error()`)
- Import order is load-bearing to avoid circular imports

**Concerns:**

None observed - clean leaf module with no external dependencies.

**Confidence:** High - Well-organized contract definitions with clear type boundaries

---

## 3. Core Infrastructure Subsystem

**Location:** `src/elspeth/core/`

**Responsibility:** Provide foundation services: configuration, canonical serialization, DAG validation, payload storage, logging, and sub-subsystems (landscape, checkpoint, retention, rate_limit, security).

**Key Components:**

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports core infrastructure |
| `canonical.py` | RFC 8785 two-phase JSON canonicalization |
| `config.py` | Dynaconf + Pydantic settings loading |
| `dag.py` | NetworkX-based ExecutionGraph |
| `payload_store.py` | FilesystemPayloadStore for large blob storage |
| `logging.py` | Structlog configuration |

**Sub-Subsystems:**

| Directory | Purpose |
|-----------|---------|
| `landscape/` | Audit trail database (see separate entry) |
| `checkpoint/` | Crash recovery (CheckpointManager, RecoveryManager) |
| `retention/` | Payload purging (PurgeManager) |
| `rate_limit/` | API throttling (RateLimiter, pyrate-limiter) |
| `security/` | Secret fingerprinting (HMAC-based) |

**Public API:**

```python
from elspeth.core import (
    # Canonical
    canonical_json, stable_hash, CANONICAL_VERSION,
    # Config
    ElspethSettings, load_settings,
    # DAG
    ExecutionGraph, GraphValidationError, NodeInfo,
    # Payload
    PayloadStore, FilesystemPayloadStore,
    # Checkpoint
    CheckpointManager, RecoveryManager, ResumePoint,
    # Logging
    configure_logging, get_logger,
)
```

**Dependencies:**

- **Inbound:** CLI, Engine, Plugins
- **Outbound:** Contracts

**Patterns Observed:**

- Two-phase canonical serialization: normalize (pandas/numpy→primitives) then serialize (rfc8785)
- NaN/Infinity strictly rejected (not silently converted) for audit integrity
- ExecutionGraph wraps NetworkX MultiDiGraph with domain operations
- Multi-source config precedence (CLI → suite → profile → pack defaults → system)
- Pydantic validation on settings loading

**Critical Design Decisions:**

```python
# Canonical JSON rejects non-finite floats
if math.isnan(obj) or math.isinf(obj):
    raise ValueError("Cannot canonicalize non-finite float")

# DAG uses MultiDiGraph for multiple edges between same nodes (fork)
self._graph: MultiDiGraph[str] = nx.MultiDiGraph()
```

**Concerns:**

- SQLite pragmas not properly passed through URL parsing (documented bug)
- Config validation errors could be more descriptive

**Confidence:** High - Solid infrastructure with clear responsibilities

---

## 4. Landscape Subsystem (Audit Trail)

**Location:** `src/elspeth/core/landscape/`

**Responsibility:** Audit backbone - records every operation for complete traceability. This is the source of truth for all pipeline execution.

**Key Components:**

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports all landscape components |
| `database.py` | LandscapeDB - SQLAlchemy Core connection management |
| `recorder.py` | LandscapeRecorder - High-level audit API |
| `schema.py` | SQLAlchemy table definitions |
| `lineage.py` | `explain()` queries for row lineage |
| `exporter.py` | LandscapeExporter for JSON/CSV export |
| `formatters.py` | CSVFormatter, JSONFormatter |
| `reproducibility.py` | Reproducibility grade computation |
| `row_data.py` | RowDataResult, RowDataState (graceful degradation) |
| `models.py` | Additional model definitions |
| `repositories.py` | Repository pattern implementations |

**Database Tables:**

| Table | Purpose |
|-------|---------|
| `runs` | Run metadata and status |
| `nodes` | Plugin instances in execution graph |
| `edges` | Connections between nodes |
| `rows` | Source row records |
| `tokens` | Row instances in DAG paths |
| `token_parents` | Fork/join relationships |
| `node_states` | Token processing records |
| `routing_events` | Gate routing decisions |
| `batches` | Aggregation batches |
| `batch_members` | Batch-to-token mapping |
| `batch_outputs` | Batch output records |
| `artifacts` | Output file records |
| `calls` | External API call records |
| `validation_errors` | Schema validation failures |
| `transform_errors` | Transform processing errors |

**Public API:**

```python
from elspeth.core.landscape import (
    LandscapeDB, LandscapeRecorder, LandscapeExporter,
    # Records
    Run, Node, Edge, Token, Row, NodeState, Batch, Artifact,
    # Query
    explain, LineageResult,
    # Export
    CSVFormatter, JSONFormatter,
)

# Recording example
recorder = LandscapeRecorder(db)
run = recorder.begin_run(config=config, canonical_version="sha256-rfc8785-v1")
node = recorder.register_node(run_id, plugin_name="csv", node_type=NodeType.SOURCE, ...)
row = recorder.create_row(run_id, source_node_id, row_index=0, data=data)
token = recorder.create_token(row_id)
state = recorder.begin_node_state(token_id, node_id, step_index=1, input_data=data)
recorder.complete_node_state(state_id, status="completed", output_data=result, duration_ms=5.2)
```

**Dependencies:**

- **Inbound:** Engine, CLI, TUI
- **Outbound:** Contracts, Core (canonical, config)

**Patterns Observed:**

- LandscapeRecorder wraps low-level database operations
- Discriminated union for NodeState (Open | Completed | Failed)
- Enum coercion at boundary (string → enum with validation)
- Hash storage survives payload deletion for integrity verification
- `explain()` gracefully handles purged payloads

**Critical Design Decisions:**

```python
# Crash on any audit integrity violation (Tier 1: Full Trust)
if row.output_hash is None:
    raise ValueError(f"COMPLETED state {state_id} has NULL output_hash - audit integrity violation")

# Token identity preserved through forks/joins
# row_id: stable source identity
# token_id: instance in specific DAG path
# parent_token_id: lineage linkage
```

**Concerns:**

None observed - this is the most mature subsystem.

**Confidence:** High - Comprehensive audit infrastructure with integrity guarantees

---

## 5. Engine Subsystem

**Location:** `src/elspeth/engine/`

**Responsibility:** Pipeline execution orchestration - manages run lifecycle, row processing, token management, and sink operations.

**Key Components:**

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports engine components |
| `orchestrator.py` | Orchestrator, PipelineConfig, RunResult - full run lifecycle |
| `processor.py` | RowProcessor - row-by-row processing with DAG traversal |
| `tokens.py` | TokenManager - token identity through forks/joins |
| `executors.py` | TransformExecutor, GateExecutor, SinkExecutor, AggregationExecutor |
| `coalesce_executor.py` | CoalesceExecutor for fork/join merging |
| `retry.py` | RetryManager, RetryConfig (tenacity-based) |
| `triggers.py` | Aggregation trigger evaluation |
| `spans.py` | SpanFactory - OpenTelemetry integration |
| `schema_validator.py` | Pipeline schema compatibility checking |
| `expression_parser.py` | ExpressionParser for config-driven gate conditions |
| `artifacts.py` | Artifact pipeline handling |

**Public API:**

```python
from elspeth.engine import (
    Orchestrator, PipelineConfig, RunResult,
    RowProcessor, TokenManager,
    TransformExecutor, GateExecutor, SinkExecutor, AggregationExecutor,
    RetryManager, RetryConfig, MaxRetriesExceeded,
    SpanFactory,
    ExpressionParser, ExpressionSecurityError,
)

# Execution example
db = LandscapeDB.from_url("sqlite:///audit.db")
orchestrator = Orchestrator(db)
config = PipelineConfig(source=csv_source, transforms=[t1, t2], sinks={"default": sink})
result = orchestrator.run(config, graph=graph, settings=settings)
```

**Dependencies:**

- **Inbound:** CLI
- **Outbound:** Landscape, Core (dag, config), Contracts, Plugins

**Patterns Observed:**

- Orchestrator manages full lifecycle (begin → execute → complete)
- RowProcessor uses work queue for fork handling (prevents stack overflow)
- Work queue iteration guard (MAX_WORK_QUEUE_ITERATIONS = 10,000)
- Aggregation is structural (engine buffers, not plugins)
- Config-driven gates (declarative routing via expressions)
- Retry with attempt tracking for audit trail

**Critical Design Decisions:**

```python
# Work queue pattern for DAG processing
work_queue: deque[_WorkItem] = deque([_WorkItem(token=token, start_step=0)])
while work_queue:
    item = work_queue.popleft()
    result, child_items = self._process_single_token(...)
    work_queue.extend(child_items)  # Fork children added to queue

# Plugin node_ids assigned by orchestrator, not plugins
source.node_id = source_id
for seq, transform in enumerate(transforms):
    transform.node_id = transform_id_map[seq]
```

**Concerns:**

- Resume implementation is partial (batch recovery works, row processing TODO)
- Some type: ignore comments for batch-aware transforms

**Confidence:** High - Well-structured execution engine with clear separation of concerns

---

## 6. Plugins Subsystem

**Location:** `src/elspeth/plugins/`

**Responsibility:** Extensible plugin framework for Sources, Transforms, Gates, and Sinks using pluggy.

**Key Components:**

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports plugin components |
| `protocols.py` | SourceProtocol, TransformProtocol, GateProtocol, SinkProtocol, CoalesceProtocol |
| `base.py` | BaseSource, BaseTransform, BaseGate, BaseSink abstract classes |
| `results.py` | GateResult, TransformResult, SourceRow, RoutingAction, RowOutcome |
| `context.py` | PluginContext (run_id, config, landscape) |
| `manager.py` | PluginManager (pluggy-based registration) |
| `hookspecs.py` | Hook specifications |
| `config_base.py` | PluginConfig, DataPluginConfig, TransformDataConfig |
| `sentinels.py` | Sentinel values |
| `schema_factory.py` | Schema factory for dynamic schema generation |
| `utils.py` | Plugin utilities |

**Plugin Implementations:**

| Directory | Plugins |
|-----------|---------|
| `sources/` | CSVSource, JSONSource |
| `transforms/` | PassThrough, FieldMapper, BatchStats, JSONExplode |
| `sinks/` | CSVSink, JSONSink, DatabaseSink |

**Public API:**

```python
from elspeth.plugins import (
    # Protocols
    SourceProtocol, TransformProtocol, GateProtocol, SinkProtocol,
    # Base classes
    BaseSource, BaseTransform, BaseGate, BaseSink,
    # Results
    TransformResult, GateResult, SourceRow, RoutingAction,
    # Context
    PluginContext,
    # Manager
    PluginManager, PluginSpec,
)

# Transform implementation example
class MyTransform(BaseTransform):
    name = "my_transform"
    input_schema = InputSchema
    output_schema = OutputSchema
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"

    def process(self, row: dict, ctx: PluginContext) -> TransformResult:
        return TransformResult.success({**row, "new_field": "value"})
```

**Dependencies:**

- **Inbound:** CLI, Engine
- **Outbound:** Contracts, Core (landscape - via context)

**Patterns Observed:**

- Protocols for type checking, base classes for convenience
- Lifecycle hooks: on_start, on_complete, close
- TransformResult factory methods: success(), error(), success_multi()
- is_batch_aware flag for aggregation transforms
- creates_tokens flag for deaggregation transforms
- _on_error config for transform error routing

**Critical Design Decisions:**

```python
# Plugins are system code, not user extensions
# Plugin bugs crash immediately (no silent recovery)
# This preserves audit integrity

# Base class defaults for metadata
determinism: Determinism = Determinism.DETERMINISTIC
plugin_version: str = "0.0.0"
is_batch_aware: bool = False
creates_tokens: bool = False
_on_error: str | None = None  # Error routing config
```

**Concerns:**

- Manager validation changes in progress (modified file)
- Protocol changes in progress (modified file)

**Confidence:** High - Clean plugin architecture with clear protocols and base classes

---

## 7. TUI Subsystem

**Location:** `src/elspeth/tui/`

**Responsibility:** Terminal User Interface for interactive lineage exploration using Textual.

**Key Components:**

| File | Description |
|------|-------------|
| `__init__.py` | Re-exports ExplainApp |
| `explain_app.py` | Main Textual app for lineage exploration |
| `types.py` | TUI-specific type definitions |
| `constants.py` | UI constants |
| `screens/` | Screen components (ExplainScreen) |
| `widgets/` | Widget components (LineageTree, NodeDetail) |

**Public API:**

```python
from elspeth.tui import ExplainApp

# Launch TUI
app = ExplainApp(run_id="abc123", token_id=token, row_id=row)
app.run()
```

**Dependencies:**

- **Inbound:** CLI
- **Outbound:** Landscape (for lineage queries)

**Patterns Observed:**

- Textual app with screen-based navigation
- Widget composition (tree, detail panels)
- Lazy loading of lineage data

**Concerns:**

None observed - focused UI implementation.

**Confidence:** Medium - Limited exploration of TUI internals, but structure is clear

---

## Dependency Graph Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                           CLI                                   │
│                    (Entry Point Layer)                          │
└─────────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌───────────┐      ┌───────────────┐      ┌─────┐
    │  Engine   │      │    Plugins    │      │ TUI │
    │(Execution)│      │ (Extensible)  │      │(UI) │
    └─────┬─────┘      └───────┬───────┘      └──┬──┘
          │                    │                 │
          └────────────────────┼─────────────────┘
                               │
                               ▼
                     ┌─────────────────┐
                     │   Landscape     │
                     │ (Audit Trail)   │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │      Core       │
                     │ (Infrastructure)│
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   Contracts     │
                     │  (Data Types)   │
                     └─────────────────┘
```

## Cross-Cutting Concerns

### Error Handling Strategy

| Layer | Strategy |
|-------|----------|
| Audit Trail (Tier 1) | Crash on any anomaly |
| Pipeline Data (Tier 2) | Wrap row operations, expect types |
| External Data (Tier 3) | Coerce, validate, quarantine |

### Observability

- **Logging:** structlog (structured events)
- **Tracing:** OpenTelemetry via SpanFactory
- **Metrics:** Not yet implemented

### Security

- **Secret Handling:** HMAC fingerprints (never store secrets)
- **Signing:** Optional export signing via ELSPETH_SIGNING_KEY
- **Sandboxing:** Plugins are system code (no user plugins)
