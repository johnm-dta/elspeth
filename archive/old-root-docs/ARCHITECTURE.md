# C4 Architecture Diagrams: ELSPETH

This document presents the ELSPETH architecture using the C4 model:
- **Level 1**: System Context
- **Level 2**: Container (Subsystem)
- **Level 3**: Component (Module)

All diagrams use Mermaid syntax for version-control-friendly rendering.

---

## Level 1: System Context Diagram

Shows ELSPETH in relation to external actors and systems.

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

**Narrative:**
- **Pipeline Operators** configure pipelines via YAML and execute them via CLI
- **Auditors** use the CLI/TUI to query lineage and verify decisions
- ELSPETH reads from various **data sources** (CSV, JSON, APIs)
- ELSPETH writes to **destinations** (files, databases)
- **LLM integration** is planned for Phase 6

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

**Container Descriptions:**

| Container | Technology | Responsibility |
|-----------|------------|----------------|
| CLI | Typer | User commands: run, explain, validate, purge, resume |
| TUI | Textual | Interactive lineage exploration |
| Engine | Python | Run lifecycle, row processing, DAG execution |
| Plugins | pluggy | Extensible sources, transforms, gates, sinks |
| Landscape | SQLAlchemy Core | Audit recording and querying |
| Core | Python | Config, canonical JSON, DAG, checkpoint, rate limit |
| Contracts | Python | Shared dataclasses, enums, protocols |
| Audit DB | SQLite/PostgreSQL | Complete audit trail storage |
| Payload Store | Filesystem | Large blob storage with retention |

---

## Level 3: Component Diagrams

### 3.1 Engine Components

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

**Component Responsibilities:**

| Component | Responsibility |
|-----------|----------------|
| Orchestrator | Begin run → register nodes/edges → process rows → complete run |
| RowProcessor | Work queue-based DAG traversal, fork/join handling |
| TokenManager | Create, fork, coalesce, expand tokens |
| Executors | Execute transforms, gates, sinks, aggregations |
| RetryManager | Retry transient failures with exponential backoff |
| SpanFactory | Create OpenTelemetry spans for observability |
| Triggers | Evaluate count/timeout triggers for aggregation |
| ExpressionParser | Parse and evaluate gate condition expressions |

### 3.2 Landscape Components

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

**Tables Managed:**

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

---

## Data Flow Diagrams

### Pipeline Execution Flow

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

### Token Lifecycle

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

---

## Trust Boundary Diagram

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

---

## Summary

This C4 documentation provides:

1. **Context**: How ELSPETH fits in the broader system landscape
2. **Containers**: The 7 major subsystems and their relationships
3. **Components**: Internal structure of Engine, Landscape, and Plugins
4. **Data Flow**: How rows flow through the pipeline with audit recording
5. **Token Lifecycle**: State transitions for row processing
6. **Deployment**: Typical development and production configurations
7. **Trust Boundaries**: The three-tier data trust model

All diagrams are in Mermaid format for easy maintenance and version control.
