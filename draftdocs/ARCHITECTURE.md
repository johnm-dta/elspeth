# ELSPETH Architecture

C4 model documentation for the ELSPETH auditable pipeline framework.

---

## At a Glance

| Question | Answer |
|----------|--------|
| **What is ELSPETH?** | Auditable Sense/Decide/Act pipeline framework |
| **Core subsystems?** | CLI, Engine, Plugins, Landscape (audit), Core |
| **Data flow?** | Source → Transforms/Gates → Sinks (all recorded) |
| **Audit storage?** | SQLite (dev) / PostgreSQL (prod) |
| **Extension model?** | pluggy-based plugin system |

---

## How to Read This Document

| Audience | Start Here |
|----------|------------|
| **New developers** | [System Context](#level-1-system-context-diagram) → [Container Diagram](#level-2-container-diagram) |
| **Plugin authors** | [Plugins Components](#33-plugins-components) |
| **Engine contributors** | [Engine Components](#31-engine-components) → [Pipeline Execution Flow](#pipeline-execution-flow) |
| **Operators** | [Deployment View](#deployment-view) |

---

## Table of Contents

- [Level 1: System Context](#level-1-system-context-diagram)
- [Level 2: Container Diagram](#level-2-container-diagram)
- [Level 3: Component Diagrams](#level-3-component-diagrams)
  - [Engine Components](#31-engine-components)
  - [Landscape Components](#32-landscape-components)
  - [Plugins Components](#33-plugins-components)
- [Data Flow Diagrams](#data-flow-diagrams)
  - [Pipeline Execution Flow](#pipeline-execution-flow)
  - [Token Lifecycle](#token-lifecycle)
- [Deployment View](#deployment-view)
- [Trust Boundary Diagram](#trust-boundary-diagram)
- [Summary](#summary)

---

## Level 1: System Context Diagram

Shows ELSPETH's relationship with external actors and systems.

```mermaid
C4Context
    title ELSPETH System Context

    Person(operator, "Pipeline Operator", "Configures and runs data pipelines")
    Person(auditor, "Auditor", "Queries lineage and verifies decisions")

    System(elspeth, "ELSPETH", "Auditable Sense/Decide/Act pipeline framework")

    System_Ext(datasources, "Data Sources", "CSV, JSON, APIs, databases")
    System_Ext(destinations, "Data Destinations", "Files, databases, message queues")
    System_Ext(llm, "LLM Providers", "OpenAI, Anthropic, etc. (Phase 6)")

    Rel(operator, elspeth, "Configures and executes pipelines", "CLI/YAML")
    Rel(auditor, elspeth, "Queries lineage", "CLI/TUI")
    Rel(elspeth, datasources, "Reads data from", "Various protocols")
    Rel(elspeth, destinations, "Writes data to", "Various protocols")
    Rel(elspeth, llm, "Calls for decisions", "HTTP/API (Phase 6)")
```

**Key relationships:**

| Actor/System | Interaction |
|--------------|-------------|
| Pipeline Operator | Configures YAML, executes via CLI, monitors runs |
| Auditor | Queries lineage via CLI/TUI, verifies decisions |
| Data Sources | CSV, JSON, APIs - read by Source plugins |
| Data Destinations | Files, databases - written by Sink plugins |
| LLM Providers | External calls for classification (Phase 6) |

---

## Level 2: Container Diagram

Shows the major subsystems within ELSPETH.

```mermaid
C4Container
    title ELSPETH Container Diagram

    Person(operator, "Operator")
    Person(auditor, "Auditor")

    Container_Boundary(elspeth, "ELSPETH Framework") {
        Container(cli, "CLI", "Typer", "Command-line interface for run, explain, validate")
        Container(tui, "TUI", "Textual", "Interactive terminal UI for lineage exploration")
        Container(engine, "Engine", "Python", "Pipeline orchestration and row processing")
        Container(plugins, "Plugins", "pluggy", "Extensible sources, transforms, sinks")
        Container(landscape, "Landscape", "SQLAlchemy Core", "Audit trail database")
        Container(core, "Core", "Python", "Configuration, canonical, DAG, checkpoint")
        Container(contracts, "Contracts", "Python", "Shared data types and protocols")
    }

    ContainerDb(auditdb, "Audit Database", "SQLite/PostgreSQL", "Stores complete audit trail")
    ContainerDb(payloads, "Payload Store", "Filesystem", "Large blob storage")

    Rel(operator, cli, "Executes pipelines")
    Rel(auditor, tui, "Explores lineage")
    Rel(auditor, cli, "Queries lineage")

    Rel(cli, engine, "Orchestrates runs")
    Rel(cli, plugins, "Instantiates plugins")
    Rel(cli, tui, "Launches")
    Rel(engine, landscape, "Records audit trail")
    Rel(engine, plugins, "Executes plugins")
    Rel(tui, landscape, "Queries lineage")
    Rel(landscape, core, "Uses canonical/config")
    Rel(landscape, auditdb, "Persists to")
    Rel(core, payloads, "Stores blobs")
    Rel(plugins, contracts, "Uses types")
    Rel(engine, contracts, "Uses types")
    Rel(core, contracts, "Uses types")
```

### Container Responsibilities

| Container | Technology | Purpose |
|-----------|------------|---------|
| **CLI** | Typer | User commands: `run`, `explain`, `validate`, `resume` |
| **TUI** | Textual | Interactive lineage exploration |
| **Engine** | Python | Run lifecycle, row processing, DAG execution |
| **Plugins** | pluggy | Extensible sources, transforms, gates, sinks |
| **Landscape** | SQLAlchemy Core | Audit recording and querying |
| **Core** | Python | Config, canonical JSON, DAG, checkpoint, rate limit |
| **Contracts** | Python | Shared dataclasses, enums, protocols |
| **Audit DB** | SQLite/PostgreSQL | Complete audit trail storage |
| **Payload Store** | Filesystem | Large blob storage with retention |

---

## Level 3: Component Diagrams

### 3.1 Engine Components

The Engine orchestrates pipeline execution and row processing.

```mermaid
C4Component
    title Engine Component Diagram

    Container_Boundary(engine, "Engine Subsystem") {
        Component(orchestrator, "Orchestrator", "Python Class", "Full run lifecycle management")
        Component(processor, "RowProcessor", "Python Class", "Row-by-row DAG traversal")
        Component(tokens, "TokenManager", "Python Class", "Token identity through forks/joins")
        Component(executors, "Executors", "Python Classes", "Transform, Gate, Sink, Aggregation execution")
        Component(retry, "RetryManager", "tenacity", "Retry logic with backoff")
        Component(spans, "SpanFactory", "OpenTelemetry", "Tracing integration")
        Component(triggers, "Triggers", "Python", "Aggregation trigger evaluation")
        Component(expression, "ExpressionParser", "Python", "Config gate condition parsing")
    }

    Rel(orchestrator, processor, "Creates and uses")
    Rel(processor, tokens, "Manages tokens via")
    Rel(processor, executors, "Delegates to")
    Rel(executors, retry, "Uses for transient failures")
    Rel(orchestrator, spans, "Creates tracing spans")
    Rel(processor, triggers, "Evaluates aggregation via")
    Rel(processor, expression, "Parses gate conditions")
```

| Component | Responsibility |
|-----------|----------------|
| **Orchestrator** | Begin run → register nodes/edges → process rows → complete run |
| **RowProcessor** | Work queue-based DAG traversal, fork/join handling |
| **TokenManager** | Create, fork, coalesce, expand tokens |
| **Executors** | Execute transforms, gates, sinks, aggregations |
| **RetryManager** | Retry transient failures with exponential backoff |
| **SpanFactory** | Create OpenTelemetry spans for observability |
| **Triggers** | Evaluate count/timeout triggers for aggregation |
| **ExpressionParser** | Parse and evaluate gate condition expressions |

### 3.2 Landscape Components

The Landscape records and queries the audit trail.

```mermaid
C4Component
    title Landscape Component Diagram

    Container_Boundary(landscape, "Landscape Subsystem") {
        Component(recorder, "LandscapeRecorder", "Python Class", "High-level audit API")
        Component(database, "LandscapeDB", "SQLAlchemy Core", "Connection management")
        Component(schema, "Schema", "SQLAlchemy Tables", "Table definitions")
        Component(lineage, "Lineage", "Python", "explain() query implementation")
        Component(exporter, "Exporter", "Python Class", "JSON/CSV export")
        Component(formatters, "Formatters", "Python Classes", "CSV/JSON formatting")
        Component(reproducibility, "Reproducibility", "Python", "Grade computation")
    }

    ContainerDb_Ext(db, "SQLite/PostgreSQL")

    Rel(recorder, database, "Uses for operations")
    Rel(recorder, schema, "Inserts/updates via")
    Rel(database, db, "Connects to")
    Rel(lineage, recorder, "Queries via")
    Rel(exporter, recorder, "Reads from")
    Rel(exporter, formatters, "Uses for output")
    Rel(recorder, reproducibility, "Computes grade via")
```

### Audit Trail Tables

```
runs → nodes → edges
  ↓
rows → tokens → token_parents
         ↓
    node_states → routing_events
         ↓           ↓
      calls     batches → batch_members
                   ↓
              batch_outputs
                   ↓
               artifacts

validation_errors, transform_errors (error tracking)
```

### 3.3 Plugins Components

The plugin system provides extensible pipeline components.

```mermaid
C4Component
    title Plugins Component Diagram

    Container_Boundary(plugins, "Plugins Subsystem") {
        Component(protocols, "Protocols", "Python", "SourceProtocol, TransformProtocol, etc.")
        Component(base, "Base Classes", "Python ABC", "BaseSource, BaseTransform, etc.")
        Component(results, "Results", "Python", "TransformResult, GateResult, etc.")
        Component(context, "PluginContext", "Python", "Runtime context for plugins")
        Component(manager, "PluginManager", "pluggy", "Discovery and registration")
        Component(hookspecs, "Hookspecs", "pluggy", "Hook specifications")
    }

    Container_Boundary(sources, "Sources") {
        Component(csv_source, "CSVSource", "Python", "Load from CSV")
        Component(json_source, "JSONSource", "Python", "Load from JSON/JSONL")
    }

    Container_Boundary(transforms, "Transforms") {
        Component(passthrough, "PassThrough", "Python", "Identity transform")
        Component(field_mapper, "FieldMapper", "Python", "Rename/select fields")
        Component(batch_stats, "BatchStats", "Python", "Aggregation statistics")
        Component(json_explode, "JSONExplode", "Python", "Deaggregation")
    }

    Container_Boundary(sinks, "Sinks") {
        Component(csv_sink, "CSVSink", "Python", "Write to CSV")
        Component(json_sink, "JSONSink", "Python", "Write to JSON/JSONL")
        Component(db_sink, "DatabaseSink", "Python", "Write to database")
    }

    Rel(base, protocols, "Implements")
    Rel(csv_source, base, "Extends BaseSource")
    Rel(json_source, base, "Extends BaseSource")
    Rel(passthrough, base, "Extends BaseTransform")
    Rel(field_mapper, base, "Extends BaseTransform")
    Rel(batch_stats, base, "Extends BaseTransform")
    Rel(json_explode, base, "Extends BaseTransform")
    Rel(csv_sink, base, "Extends BaseSink")
    Rel(json_sink, base, "Extends BaseSink")
    Rel(db_sink, base, "Extends BaseSink")
```

| Component | Purpose |
|-----------|---------|
| **Protocols** | Runtime-checkable interfaces (`SourceProtocol`, `TransformProtocol`, etc.) |
| **Base Classes** | Abstract implementations with common functionality |
| **Results** | Typed results (`TransformResult`, `GateResult`, `SourceRow`) |
| **PluginContext** | Runtime context passed to all plugin methods |
| **PluginManager** | pluggy-based discovery and registration |

---

## Data Flow Diagrams

### Pipeline Execution Flow

This sequence shows how a row flows through the pipeline with audit recording at each step.

```mermaid
sequenceDiagram
    participant CLI
    participant Orchestrator
    participant Recorder as LandscapeRecorder
    participant Processor as RowProcessor
    participant Source
    participant Transform
    participant Sink

    CLI->>Orchestrator: run(PipelineConfig)
    Orchestrator->>Recorder: begin_run(config)
    Recorder-->>Orchestrator: Run

    loop For each node
        Orchestrator->>Recorder: register_node(...)
    end

    loop For each edge
        Orchestrator->>Recorder: register_edge(...)
    end

    Orchestrator->>Source: load(ctx)

    loop For each row
        Source-->>Orchestrator: SourceRow
        Orchestrator->>Processor: process_row(row_data)
        Processor->>Recorder: create_row(...)
        Processor->>Recorder: create_token(...)

        loop For each transform
            Processor->>Recorder: begin_node_state(...)
            Processor->>Transform: process(row, ctx)
            Transform-->>Processor: TransformResult
            Processor->>Recorder: complete_node_state(...)
        end

        Processor-->>Orchestrator: RowResult
    end

    loop For each sink
        Orchestrator->>Sink: write(rows, ctx)
        Sink-->>Orchestrator: ArtifactDescriptor
        Orchestrator->>Recorder: register_artifact(...)
    end

    Orchestrator->>Recorder: complete_run(status)
    Orchestrator-->>CLI: RunResult
```

**Key audit points:**

1. `begin_run` - Configuration hash stored
2. `register_node/edge` - DAG structure recorded
3. `create_row/token` - Row identity established
4. `begin/complete_node_state` - Transform input/output hashes recorded
5. `register_artifact` - Sink output hash recorded
6. `complete_run` - Final status and timestamps

### Token Lifecycle

Tokens track row identity through forks, joins, and routing decisions.

```mermaid
stateDiagram-v2
    [*] --> Created: Source yields row
    Created --> Processing: Enter transform chain

    state Processing {
        [*] --> Transform
        Transform --> Transform: Continue
        Transform --> Gate: Route decision

        Gate --> Forked: fork_to_paths
        Gate --> Routed: route_to_sink
        Gate --> Transform: continue

        state Forked {
            [*] --> Child1
            [*] --> Child2
            Child1 --> Processing
            Child2 --> Processing
        }
    }

    Processing --> Completed: Reach output sink
    Processing --> Routed: Gate routes to sink
    Processing --> Quarantined: Validation failure
    Processing --> Failed: Processing error
    Processing --> ConsumedInBatch: Aggregated
    Forked --> Coalesced: Merge point

    Completed --> [*]
    Routed --> [*]
    Quarantined --> [*]
    Failed --> [*]
    ConsumedInBatch --> [*]
    Coalesced --> [*]
```

**Terminal states:**

| State | Meaning |
|-------|---------|
| `COMPLETED` | Reached output sink |
| `ROUTED` | Gate sent to named sink |
| `FORKED` | Split to multiple paths (parent token) |
| `CONSUMED_IN_BATCH` | Aggregated into batch |
| `COALESCED` | Merged at join point |
| `QUARANTINED` | Failed validation, stored for investigation |
| `FAILED` | Processing error, not recoverable |

---

## Deployment View

```mermaid
C4Deployment
    title ELSPETH Deployment (Typical)

    Deployment_Node(dev, "Developer Machine") {
        Deployment_Node(venv, "Python venv") {
            Container(cli_inst, "elspeth CLI", "Python", "Pipeline execution")
        }
        ContainerDb(sqlite, "landscape.db", "SQLite", "Local audit trail")
        Container(payloads_dir, "payloads/", "Filesystem", "Payload storage")
    }

    Deployment_Node(prod, "Production Server") {
        Deployment_Node(container, "Docker Container") {
            Container(cli_prod, "elspeth CLI", "Python", "Pipeline execution")
        }
        ContainerDb(postgres, "PostgreSQL", "PostgreSQL", "Shared audit trail")
        Container(blob_store, "Blob Storage", "S3/Azure", "Payload storage")
    }

    Rel(cli_inst, sqlite, "Writes to")
    Rel(cli_inst, payloads_dir, "Stores payloads")
    Rel(cli_prod, postgres, "Writes to")
    Rel(cli_prod, blob_store, "Stores payloads")
```

| Environment | Audit DB | Payload Store |
|-------------|----------|---------------|
| Development | SQLite (`landscape.db`) | Local filesystem |
| Production | PostgreSQL | S3/Azure Blob Storage |

---

## Trust Boundary Diagram

The Three-Tier Trust Model defines how data is handled at each boundary.

```mermaid
flowchart TB
    subgraph TIER1["TIER 1: Our Data (Full Trust)"]
        direction TB
        AuditDB[(Audit Database)]
        Landscape[Landscape Recorder]

        Landscape --> AuditDB
        note1["Crash on any anomaly<br/>No coercion ever"]
    end

    subgraph TIER2["TIER 2: Pipeline Data (Elevated Trust)"]
        direction TB
        Transforms[Transforms]
        Gates[Gates]
        Sinks[Sinks]

        note2["Types valid but values can fail<br/>Wrap operations on row values<br/>No type coercion"]
    end

    subgraph TIER3["TIER 3: External Data (Zero Trust)"]
        direction TB
        Sources[Sources]
        APIs[External APIs]

        note3["Coerce where possible<br/>Validate at boundary<br/>Quarantine failures"]
    end

    APIs --> Sources
    Sources --> Transforms
    Transforms --> Gates
    Gates --> Sinks
    Sources --> Landscape
    Transforms --> Landscape
    Gates --> Landscape
    Sinks --> Landscape

    style TIER1 fill:#d4edda
    style TIER2 fill:#fff3cd
    style TIER3 fill:#f8d7da
```

### Trust Tier Summary

| Tier | Trust Level | Coercion | On Error |
|------|-------------|----------|----------|
| **Tier 1** (Audit DB) | Full trust | Never | Crash immediately |
| **Tier 2** (Pipeline) | Elevated ("probably OK") | Never | Return error result |
| **Tier 3** (External) | Zero trust | At boundary | Quarantine row |

---

## Summary

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **SQLAlchemy Core** (not ORM) | Audit trail needs precise SQL, not object mapping |
| **pluggy** | Battle-tested (pytest), clean hook system |
| **Canonical JSON** | Deterministic hashing for audit integrity |
| **Token-based lineage** | Tracks identity through forks/joins |
| **Three-tier trust** | Clear rules for coercion and error handling |

### What This Document Covers

1. **Context** - How ELSPETH fits in the system landscape
2. **Containers** - 7 major subsystems and relationships
3. **Components** - Internal structure of Engine, Landscape, Plugins
4. **Data Flow** - How rows flow with audit recording
5. **Token Lifecycle** - State transitions for row processing
6. **Deployment** - Development and production configurations
7. **Trust Boundaries** - Three-tier data trust model

All diagrams use Mermaid syntax for version control compatibility.

---

## See Also

- [README.md](README.md) - Project overview and quick start
- [PLUGIN.md](PLUGIN.md) - Plugin development guide
- [CLAUDE.md](CLAUDE.md) - Complete project context and patterns
- [docs/contracts/](docs/contracts/) - Protocol specifications
