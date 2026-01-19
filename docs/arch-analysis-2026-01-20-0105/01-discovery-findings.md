# Discovery Findings: ELSPETH Architecture Analysis

## Executive Summary

ELSPETH is a **domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines** designed for high-stakes accountability. The framework provides scaffolding for data processing workflows where every decision must be traceable to its source—whether the "decide" step is an LLM, ML model, rules engine, or threshold check.

**Key Insight**: This is an audit-first framework. The Landscape audit trail is the source of truth, and the philosophy is "if it's not recorded, it didn't happen."

## Project Metadata

| Attribute | Value |
|-----------|-------|
| Version | 0.1.0 |
| Status | Approaching RC-1 (LLM integration Phase 6 of 7) |
| Python Files | 88 |
| Lines of Code | ~10,600 |
| Primary Entry Point | `cli.py` (Typer CLI) |
| Database | SQLAlchemy Core (SQLite default) |
| Test Framework | pytest |

## Architecture Model: Sense/Decide/Act (SDA)

```
SENSE (Sources) → DECIDE (Transforms/Gates) → ACT (Sinks)
```

- **Source**: Load data (CSV, JSON, API, database) - exactly 1 per run
- **Transform**: Process/classify data - 0+ ordered, includes Gates for routing
- **Gate**: Evaluate row → decide destination(s) via continue, route_to_sink, or fork_to_paths
- **Aggregation**: Collect N rows until trigger → emit result (stateful)
- **Coalesce**: Merge results from parallel paths
- **Sink**: Output results - 1+ named destinations

## Directory Structure

```
src/elspeth/
├── __init__.py           # Package version (0.1.0)
├── cli.py                # Typer CLI - run, explain, validate, purge, resume
├── contracts/            # Cross-boundary data types (audit records, results, enums)
│   ├── audit.py          # Run, Node, Edge, Token, Row, etc.
│   ├── enums.py          # RunStatus, RowOutcome, NodeType, etc.
│   ├── results.py        # TransformResult, GateResult, SourceRow
│   ├── routing.py        # RoutingAction, RoutingSpec, EdgeInfo
│   ├── config.py         # Settings dataclasses
│   └── ...
├── core/                 # Infrastructure
│   ├── canonical.py      # RFC 8785 deterministic JSON serialization
│   ├── config.py         # Dynaconf + Pydantic settings
│   ├── dag.py            # NetworkX execution graph
│   ├── payload_store.py  # Large blob storage
│   ├── logging.py        # Structlog configuration
│   ├── landscape/        # Audit trail database
│   ├── checkpoint/       # Crash recovery
│   ├── retention/        # Payload purging
│   ├── rate_limit/       # API throttling
│   └── security/         # Secret fingerprinting
├── engine/               # Pipeline execution
│   ├── orchestrator.py   # Full run lifecycle
│   ├── processor.py      # Row-by-row processing
│   ├── tokens.py         # Token identity management
│   ├── executors.py      # Transform/Gate/Sink executors
│   ├── retry.py          # Retry with tenacity
│   ├── triggers.py       # Aggregation triggers
│   └── spans.py          # OpenTelemetry integration
├── plugins/              # Plugin framework
│   ├── protocols.py      # SourceProtocol, TransformProtocol, etc.
│   ├── base.py           # BaseSource, BaseTransform, etc.
│   ├── manager.py        # pluggy-based registration
│   ├── hookspecs.py      # Hook specifications
│   ├── sources/          # CSV, JSON sources
│   ├── transforms/       # PassThrough, FieldMapper, BatchStats, JSONExplode
│   └── sinks/            # CSV, JSON, Database sinks
└── tui/                  # Terminal UI
    ├── explain_app.py    # Textual app for lineage exploration
    ├── screens/          # Screen components
    └── widgets/          # Lineage tree, node detail widgets
```

## Technology Stack

### Core Framework
| Component | Technology | Purpose |
|-----------|------------|---------|
| CLI | Typer | Type-safe command-line interface |
| TUI | Textual | Interactive terminal UI for explain/status |
| Configuration | Dynaconf + Pydantic | Multi-source config + validation |
| Plugins | pluggy | Extensible hook system (same as pytest uses) |
| Data | pandas | Tabular data manipulation |
| Database | SQLAlchemy Core | Multi-backend without ORM overhead |
| Migrations | Alembic | Schema versioning |
| Retries | tenacity | Industry standard backoff |

### Acceleration Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Canonical JSON | rfc8785 | RFC 8785/JCS deterministic serialization |
| DAG Validation | NetworkX | Graph algorithms (acyclicity, topo sort) |
| Observability | OpenTelemetry | Tracing integration |
| Logging | structlog | Structured event logging |
| Rate Limiting | pyrate-limiter | API throttling |

## Key Design Patterns

### 1. Three-Tier Trust Model
ELSPETH has a strict data trust hierarchy:

1. **Tier 1: Our Data (Audit Database)** - FULL TRUST
   - Bad data in audit trail = crash immediately
   - No coercion, no defaults, no silent recovery

2. **Tier 2: Pipeline Data (Post-Source)** - ELEVATED TRUST
   - Type-valid but potentially operation-unsafe
   - No coercion at transform/sink level

3. **Tier 3: External Data (Source Input)** - ZERO TRUST
   - Coerce where possible, quarantine failures
   - Source is the ONLY place coercion is allowed

### 2. Token Identity System
Tracks row instances through DAG execution:
- `row_id`: Stable source row identity
- `token_id`: Instance of row in specific DAG path
- `parent_token_id`: Lineage for forks and joins

### 3. Terminal Row States
Every row reaches exactly one terminal state:
- `COMPLETED` - Reached output sink
- `ROUTED` - Sent to named sink by gate
- `FORKED` - Split to multiple paths (parent token)
- `CONSUMED_IN_BATCH` - Aggregated into batch
- `COALESCED` - Merged in join
- `QUARANTINED` - Failed, stored for investigation
- `FAILED` - Failed, not recoverable

### 4. Two-Phase Canonical JSON
Deterministic hashing for audit integrity:
1. **Phase 1 (ELSPETH)**: Normalize pandas/numpy → JSON primitives, reject NaN/Infinity
2. **Phase 2 (rfc8785)**: RFC 8785/JCS standard serialization

### 5. System-Owned Plugins
All plugins are system code, not user extensions. This means:
- Plugin bugs crash immediately (no silent recovery)
- Plugins are tested with same rigor as engine code
- Users configure which plugins to use, not write their own

## Subsystem Dependencies

```
                    ┌─────────────┐
                    │     CLI     │
                    └──────┬──────┘
                           │ uses
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
    ┌───────────┐  ┌───────────────┐  ┌─────┐
    │  Engine   │  │    Plugins    │  │ TUI │
    └─────┬─────┘  └───────┬───────┘  └──┬──┘
          │                │             │
          │    uses        │    uses     │
          │    ┌───────────┼─────────────┘
          │    │           │
          ▼    ▼           │
    ┌─────────────┐        │
    │  Landscape  │◄───────┘
    └──────┬──────┘
           │ builds on
           ▼
    ┌─────────────┐
    │    Core     │
    │  (config,   │
    │  canonical, │
    │    dag)     │
    └──────┬──────┘
           │ references
           ▼
    ┌─────────────┐
    │  Contracts  │
    └─────────────┘
```

## Data Flow Through System

```
1. CLI loads settings (Dynaconf + Pydantic validation)
2. ExecutionGraph built from config (NetworkX DAG)
3. Orchestrator begins run (LandscapeRecorder.begin_run)
4. Nodes/Edges registered in Landscape
5. Source loads rows → yields SourceRow
6. For each row:
   a. TokenManager creates initial token
   b. RowProcessor executes transforms
   c. Gates evaluate → route/continue/fork
   d. Aggregations buffer → flush on trigger
   e. Results buffered for batch sink write
7. SinkExecutor writes batches → ArtifactDescriptor
8. Orchestrator completes run (status, export)
```

## Notable Architectural Decisions

1. **No ORM**: SQLAlchemy Core only - explicit SQL for performance and auditability
2. **No Legacy Code Policy**: When something changes, delete old code completely
3. **No Defensive Programming**: Crash on bugs, don't hide them
4. **Aggregation is Structural**: Engine buffers rows, not plugins (clean separation)
5. **Config-Driven Gates**: Routing is declarative, not coded in plugins

## Identified Concerns

1. **Plugin Registration**: Currently hardcoded in CLI, no dynamic discovery
2. **Resume Implementation**: Partial - validation works, actual processing TODO
3. **SQLite Pragmas**: Not properly passed through SQLAlchemy URL parsing
4. **Coalesce Config**: May be ignored in some code paths

## Entry Points

| Entry Point | Purpose |
|-------------|---------|
| `elspeth run` | Execute pipeline with audit trail |
| `elspeth explain` | Query lineage for row/token |
| `elspeth validate` | Check config without running |
| `elspeth purge` | Delete old payloads |
| `elspeth resume` | Resume failed run from checkpoint |
| `elspeth plugins list` | List available plugins |

## Files by Modification Date (Recent Activity)

Based on git status, recent work areas:
- `docs/plans/2026-01-19-multi-row-output.md` - Multi-row transform output
- `src/elspeth/plugins/manager.py` - Plugin management
- `src/elspeth/plugins/protocols.py` - Protocol definitions
- `src/elspeth/plugins/sinks/csv_sink.py` - CSV sink implementation

## Next Steps

This discovery provides the foundation for:
1. **Subsystem Catalog** - Detailed documentation of each component
2. **C4 Diagrams** - Visual architecture at multiple levels
3. **Quality Assessment** - Code health evaluation
4. **Architect Handover** - Actionable improvement recommendations
