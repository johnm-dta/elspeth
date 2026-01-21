# Architecture Diagrams

**Analysis Date:** 2026-01-21
**Analyst:** Claude Code (Opus 4.5)

This document extends the existing ARCHITECTURE.md with additional diagrams derived from the deep analysis.

---

## 1. Enhanced Container Diagram

The existing ARCHITECTURE.md container diagram is accurate. This enhanced version adds the Production Operations subsystem and shows more specific data flows.

```mermaid
C4Container
    title ELSPETH Container Diagram (Enhanced)

    Person(operator, "Pipeline Operator")
    Person(auditor, "Auditor")

    Container_Boundary(elspeth, "ELSPETH Framework") {
        Container(cli, "CLI", "Typer", "run, explain, validate, purge, resume, health")
        Container(tui, "TUI", "Textual", "Interactive lineage explorer")
        Container(engine, "Engine", "Python", "Orchestrator, Processor, Executors")
        Container(plugins, "Plugin System", "pluggy", "Protocols, Base classes, Discovery")
        Container(plugin_impl, "Plugin Implementations", "Python", "Sources, Transforms, Sinks, LLM, Azure")
        Container(landscape, "Landscape", "SQLAlchemy Core", "Audit recording (16 tables)")
        Container(core, "Core Utilities", "Python", "Canonical, Config, DAG, Logging")
        Container(prod_ops, "Production Ops", "Python", "Checkpoint, Retention, Rate Limit, Security")
        Container(contracts, "Contracts", "Python", "60+ shared types")
    }

    ContainerDb(auditdb, "Audit Database", "SQLite/PostgreSQL", "Complete audit trail")
    ContainerDb(payloads, "Payload Store", "Filesystem", "Content-addressable blobs")
    System_Ext(llm, "LLM Providers", "Azure OpenAI, OpenRouter")
    System_Ext(azure, "Azure Services", "Blob Storage, Content Safety")

    Rel(operator, cli, "Executes pipelines")
    Rel(auditor, tui, "Explores lineage")
    Rel(cli, engine, "Orchestrates")
    Rel(cli, prod_ops, "Uses for resume/purge")
    Rel(engine, landscape, "Records audit")
    Rel(engine, plugins, "Uses protocols")
    Rel(plugins, plugin_impl, "Discovers")
    Rel(plugin_impl, llm, "Calls")
    Rel(plugin_impl, azure, "Calls")
    Rel(landscape, auditdb, "Persists")
    Rel(core, payloads, "Stores")
    Rel(prod_ops, landscape, "Reads/writes")
    Rel(prod_ops, payloads, "Purges")
```

---

## 2. Dependency Flow Diagram

Shows the direction of dependencies between subsystems.

```mermaid
flowchart TD
    subgraph Foundation["Foundation Layer"]
        Contracts[Contracts<br/>~2K LOC]
    end

    subgraph Core["Core Infrastructure"]
        CoreUtils[Core Utilities<br/>~2K LOC]
        Landscape[Landscape<br/>~4.6K LOC]
        ProdOps[Production Ops<br/>~1.2K LOC]
    end

    subgraph Runtime["Runtime Layer"]
        PluginSys[Plugin System<br/>~2.3K LOC]
        Engine[Engine<br/>~5.9K LOC]
    end

    subgraph Extensions["Extension Layer"]
        PluginImpl[Plugin Implementations<br/>~7K LOC]
    end

    subgraph Interface["Interface Layer"]
        CLI[CLI/TUI<br/>~2K LOC]
    end

    Contracts --> CoreUtils
    Contracts --> Landscape
    Contracts --> PluginSys
    Contracts --> Engine
    Contracts --> ProdOps

    CoreUtils --> Landscape
    CoreUtils --> Engine
    CoreUtils --> PluginSys
    CoreUtils --> ProdOps

    Landscape --> Engine
    Landscape --> ProdOps
    Landscape --> CLI

    PluginSys --> Engine
    PluginSys --> PluginImpl

    Engine --> CLI

    ProdOps --> CLI

    PluginImpl --> Engine

    style Foundation fill:#e1f5fe
    style Core fill:#fff3e0
    style Runtime fill:#e8f5e9
    style Extensions fill:#f3e5f5
    style Interface fill:#fce4ec
```

---

## 3. Data Flow Through Pipeline

Shows how data moves through the SDA pipeline with audit recording.

```mermaid
sequenceDiagram
    participant Source
    participant Engine
    participant Landscape as Landscape DB
    participant PayloadStore as Payload Store
    participant Transform
    participant Gate
    participant Sink

    Note over Source,Sink: SENSE Phase

    Source->>Engine: load() yields SourceRow
    Engine->>PayloadStore: store(row_data)
    PayloadStore-->>Engine: hash (content address)
    Engine->>Landscape: create_row(hash, ref)
    Engine->>Landscape: create_token(row_id)

    Note over Source,Sink: DECIDE Phase

    loop For each transform
        Engine->>Landscape: begin_node_state(token_id, node_id)
        Engine->>Transform: process(row, ctx)
        Transform-->>Engine: TransformResult
        Engine->>PayloadStore: store(output_data)
        Engine->>Landscape: complete_node_state(output_hash)
    end

    opt If Gate exists
        Engine->>Gate: evaluate(row, ctx)
        Gate-->>Engine: GateResult(RoutingAction)
        alt route_to_sink
            Engine->>Landscape: record_routing_event(edge_id)
            Engine->>Landscape: record_token_outcome(ROUTED)
        else fork_to_paths
            Engine->>Landscape: record_token_outcome(FORKED)
            loop For each path
                Engine->>Landscape: create_token(child, parent_id)
            end
        else continue
            Engine->>Landscape: record_routing_event(edge_id, CONTINUE)
        end
    end

    Note over Source,Sink: ACT Phase

    Engine->>Landscape: begin_node_state(sink_node)
    Engine->>Sink: write(tokens, ctx)
    Sink-->>Engine: ArtifactDescriptor(hash, path)
    Engine->>Landscape: complete_node_state()
    Engine->>Landscape: register_artifact(hash, path)
    Engine->>Landscape: record_token_outcome(COMPLETED)
```

---

## 4. Token State Machine (Detailed)

Enhanced state diagram showing all transitions and conditions.

```mermaid
stateDiagram-v2
    [*] --> Created: Source yields row

    Created --> InTransform: Enter pipeline

    state InTransform {
        [*] --> Processing
        Processing --> Processing: TransformResult.success()
        Processing --> Error: TransformResult.error()

        Error --> Quarantined: on_error not configured
        Error --> Routed: on_error = sink_name
    }

    InTransform --> AtGate: Reach gate node

    state AtGate {
        [*] --> Evaluating
        Evaluating --> Continue: RoutingAction.continue_()
        Evaluating --> Route: RoutingAction.route()
        Evaluating --> Fork: RoutingAction.fork_to_paths()
    }

    Continue --> InTransform: Next transform
    Route --> ROUTED: terminal

    Fork --> FORKED: Parent token terminal
    state FORKED {
        [*] --> Child1
        [*] --> Child2
        [*] --> ChildN
    }
    FORKED --> InTransform: Children continue

    InTransform --> AtAggregation: Reach aggregation

    state AtAggregation {
        [*] --> Buffering
        Buffering --> Buffering: !trigger_fired
        Buffering --> CONSUMED_IN_BATCH: trigger_fired
    }

    CONSUMED_IN_BATCH --> BatchProcessing
    BatchProcessing --> InTransform: Batch output token

    InTransform --> AtCoalesce: Reach coalesce

    state AtCoalesce {
        [*] --> Waiting
        Waiting --> Waiting: !all_branches_arrived
        Waiting --> COALESCED: policy satisfied
    }

    COALESCED --> InTransform: Merged token continues

    InTransform --> AtSink: Reach sink
    AtSink --> COMPLETED: write() succeeds
    AtSink --> FAILED: write() fails

    COMPLETED --> [*]
    ROUTED --> [*]
    FORKED --> [*]
    CONSUMED_IN_BATCH --> [*]
    COALESCED --> [*]
    Quarantined --> [*]
    FAILED --> [*]
```

---

## 5. Plugin Protocol Hierarchy

Shows the relationship between protocols, base classes, and implementations.

```mermaid
classDiagram
    class PluginProtocol {
        <<protocol>>
        +name: str
        +plugin_version: str
        +determinism: Determinism
        +node_id: str
        +on_start(ctx)
        +on_complete(ctx)
        +close()
    }

    class SourceProtocol {
        <<protocol>>
        +output_schema: PluginSchema
        +load(ctx) Iterator~SourceRow~
    }

    class TransformProtocol {
        <<protocol>>
        +input_schema: PluginSchema
        +output_schema: PluginSchema
        +is_batch_aware: bool
        +process(row, ctx) TransformResult
    }

    class GateProtocol {
        <<protocol>>
        +input_schema: PluginSchema
        +output_schema: PluginSchema
        +evaluate(row, ctx) GateResult
    }

    class SinkProtocol {
        <<protocol>>
        +input_schema: PluginSchema
        +write(rows, ctx) ArtifactDescriptor
    }

    class CoalesceProtocol {
        <<protocol>>
        +output_schema: PluginSchema
        +policy: CoalescePolicy
        +merge(outputs, ctx) dict
    }

    PluginProtocol <|-- SourceProtocol
    PluginProtocol <|-- TransformProtocol
    PluginProtocol <|-- GateProtocol
    PluginProtocol <|-- SinkProtocol
    PluginProtocol <|-- CoalesceProtocol

    class BaseSource {
        <<abstract>>
        +load(ctx)*
    }
    class BaseTransform {
        <<abstract>>
        +process(row, ctx)*
    }
    class BaseGate {
        <<abstract>>
        +evaluate(row, ctx)*
    }
    class BaseSink {
        <<abstract>>
        +write(rows, ctx)*
    }

    SourceProtocol <|.. BaseSource
    TransformProtocol <|.. BaseTransform
    GateProtocol <|.. BaseGate
    SinkProtocol <|.. BaseSink

    class CSVSource
    class JSONSource
    class AzureBlobSource
    BaseSource <|-- CSVSource
    BaseSource <|-- JSONSource
    BaseSource <|-- AzureBlobSource

    class PassThrough
    class FieldMapper
    class AzureLLMTransform
    class AzureContentSafety
    BaseTransform <|-- PassThrough
    BaseTransform <|-- FieldMapper
    BaseTransform <|-- AzureLLMTransform
    BaseTransform <|-- AzureContentSafety

    class CSVSink
    class JSONSink
    class DatabaseSink
    class AzureBlobSink
    BaseSink <|-- CSVSink
    BaseSink <|-- JSONSink
    BaseSink <|-- DatabaseSink
    BaseSink <|-- AzureBlobSink
```

---

## 6. Landscape Database ER Diagram

Shows the 16 tables and their relationships.

```mermaid
erDiagram
    runs ||--o{ nodes : contains
    runs ||--o{ edges : contains
    runs ||--o{ rows : contains
    runs ||--o{ batches : contains
    runs ||--o{ validation_errors : logs
    runs ||--o{ transform_errors : logs
    runs ||--o{ checkpoints : stores
    runs ||--o{ token_outcomes : records

    nodes ||--o{ node_states : tracks
    nodes ||--o{ artifacts : produces

    rows ||--o{ tokens : creates

    tokens ||--o{ node_states : processes
    tokens ||--o{ token_parents : has_parent
    tokens ||--o{ batch_members : joins
    tokens ||--o{ token_outcomes : has
    tokens }o--|| tokens : parent_of

    node_states ||--o{ calls : makes
    node_states ||--o{ routing_events : decides
    node_states ||--o{ artifacts : produces

    edges ||--o{ routing_events : uses

    batches ||--o{ batch_members : contains
    batches ||--o{ batch_outputs : produces

    runs {
        string run_id PK
        string config_hash
        string status
        datetime started_at
        datetime completed_at
        string reproducibility_grade
        string export_status
    }

    nodes {
        string node_id PK
        string run_id FK
        string plugin_name
        string plugin_version
        string node_type
        string determinism
        string config_hash
        string schema_hash
    }

    tokens {
        string token_id PK
        string row_id FK
        string fork_group_id
        string join_group_id
        string expand_group_id
        string branch_name
    }

    node_states {
        string state_id PK
        string token_id FK
        string node_id FK
        int step_index
        int attempt
        string status
        string input_hash
        string output_hash
        float duration_ms
    }

    calls {
        string call_id PK
        string state_id FK
        int call_index
        string call_type
        string status
        string request_hash
        string response_hash
        float latency_ms
    }
```

---

## 7. Checkpoint/Recovery Flow

Shows the checkpoint creation and resume process.

```mermaid
flowchart TD
    subgraph NormalExecution["Normal Execution"]
        A[Process Row] --> B{Checkpoint Interval?}
        B -->|Yes| C[CheckpointManager.create_checkpoint]
        C --> D[Record token_id, node_id, sequence]
        D --> E[Serialize aggregation_state_json]
        B -->|No| F[Continue]
    end

    subgraph CrashRecovery["Crash Recovery"]
        G[CLI: elspeth resume] --> H[RecoveryManager.can_resume]
        H --> I{Run exists?}
        I -->|No| J[Cannot Resume]
        I -->|Yes| K{Status = FAILED?}
        K -->|No| J
        K -->|Yes| L{Checkpoints exist?}
        L -->|No| J
        L -->|Yes| M[get_resume_point]
        M --> N[Load latest checkpoint]
        N --> O[Deserialize aggregation state]
        O --> P[get_unprocessed_rows]
        P --> Q[Find rows after checkpoint]
        Q --> R[Fetch row_data from PayloadStore]
        R --> S[Orchestrator.resume]
        S --> T[Process unprocessed rows]
        T --> U{Success?}
        U -->|Yes| V[delete_checkpoints]
        U -->|No| W[Keep checkpoints for retry]
    end

    E --> |Crash| G
    F --> |Crash| G
```

---

## 8. Three-Tier Trust Model Detail

Shows where each tier is enforced in the codebase.

```mermaid
flowchart TB
    subgraph External["TIER 3: External Data (Zero Trust)"]
        ExtAPI[External APIs<br/>LLM, Azure, HTTP]
        ExtFile[External Files<br/>CSV, JSON]
    end

    subgraph Sources["Source Layer"]
        CSVSrc[CSVSource<br/>allow_coercion=True]
        JSONSrc[JSONSource<br/>allow_coercion=True]
        BlobSrc[AzureBlobSource<br/>allow_coercion=True]
    end

    subgraph Pipeline["TIER 2: Pipeline Data (Elevated Trust)"]
        Trans[Transforms<br/>allow_coercion=False]
        Gates[Gates<br/>allow_coercion=False]
        Sinks[Sinks<br/>allow_coercion=False]
    end

    subgraph Audit["TIER 1: Audit Data (Full Trust)"]
        Recorder[LandscapeRecorder]
        Repos[Repositories<br/>_coerce_enum fails fast]
        DB[(Audit Database)]
    end

    ExtAPI --> CSVSrc
    ExtFile --> CSVSrc
    ExtFile --> JSONSrc
    ExtAPI --> BlobSrc

    CSVSrc -->|Types validated<br/>Invalid → Quarantine| Trans
    JSONSrc -->|Types validated<br/>Invalid → Quarantine| Trans
    BlobSrc -->|Types validated<br/>Invalid → Quarantine| Trans

    Trans -->|Types trusted<br/>Wrong type = BUG| Gates
    Gates -->|Types trusted<br/>Wrong type = BUG| Sinks

    Trans -.->|Record input/output| Recorder
    Gates -.->|Record routing| Recorder
    Sinks -.->|Record artifact| Recorder

    Recorder -->|Strict insert| DB
    DB -->|CRASH on anomaly| Repos
    Repos -->|Enum coercion<br/>fails fast| Recorder

    style External fill:#f8d7da
    style Sources fill:#fff3cd
    style Pipeline fill:#fff3cd
    style Audit fill:#d4edda
```

---

## 9. LLM Transform Execution Patterns

Shows the three execution modes for LLM transforms.

```mermaid
flowchart TD
    subgraph Sequential["Sequential (pool_size=1)"]
        S1[Receive row] --> S2[Create LLM request]
        S2 --> S3[Call API]
        S3 --> S4[Record call to audit]
        S4 --> S5[Return TransformResult]
    end

    subgraph Pooled["Pooled (pool_size>1)"]
        P1[Receive row] --> P2[PooledExecutor.submit]
        P2 --> P3[Semaphore acquire]
        P3 --> P4[ThreadPool execute]
        P4 --> P5[AIMDThrottle delay]
        P5 --> P6[Call API]
        P6 --> P7{429/503/529?}
        P7 -->|Yes| P8[CapacityError<br/>Backoff retry]
        P7 -->|No| P9[Record call]
        P8 --> P5
        P9 --> P10[ReorderBuffer]
        P10 --> P11[Return in order]
    end

    subgraph Batch["Azure Batch API"]
        B1[Collect rows] --> B2{Trigger fired?}
        B2 -->|No| B1
        B2 -->|Yes| B3[Submit batch job]
        B3 --> B4[update_checkpoint<br/>job_id]
        B4 --> B5[Raise BatchPendingError]
        B5 --> B6[Engine schedules poll]
        B6 --> B7{Job complete?}
        B7 -->|No| B6
        B7 -->|Yes| B8[Retrieve results]
        B8 --> B9[clear_checkpoint]
        B9 --> B10[Return TransformResult]
    end
```

---

## 10. Configuration Loading Flow

Shows how YAML configuration becomes validated settings.

```mermaid
flowchart TD
    subgraph Input["Configuration Sources"]
        YAML[settings.yaml]
        ENV[Environment<br/>ELSPETH_*]
        DOT[.env file<br/>optional]
    end

    subgraph Loading["Dynaconf Loading"]
        DC[Dynaconf] --> |settings_files| YAML
        DC --> |envvar_prefix| ENV
        DC --> |merge_enabled| M[Deep merge]
    end

    subgraph Processing["Config Processing"]
        M --> RAW[raw_config dict]
        RAW --> EXP[_expand_env_vars<br/>\${VAR:-default}]
        EXP --> TMPL[_expand_config_templates<br/>template_file, lookup_file]
        TMPL --> FP[_fingerprint_secrets<br/>for audit copy only]
    end

    subgraph Validation["Pydantic Validation"]
        TMPL --> PYD[ElspethSettings\nmodel_config=frozen]
        PYD --> |ValidationError| ERR[Config Error]
        PYD --> |Success| VALID[Validated Settings]
    end

    subgraph Runtime["Runtime Use"]
        VALID --> CLI[CLI execution<br/>secrets available]
        FP --> AUDIT[resolve_config\nfor audit storage]
    end

    DOT -.-> |python-dotenv| ENV
```

---

## Summary

These diagrams complement the existing ARCHITECTURE.md by providing:

1. **Enhanced Container Diagram** - Includes Production Ops, LLM/Azure connections
2. **Dependency Flow** - Shows layered architecture with LOC estimates
3. **Data Flow** - Detailed sequence of audit recording
4. **Token State Machine** - All states and transitions
5. **Plugin Hierarchy** - Protocols → Base classes → Implementations
6. **Landscape ER Diagram** - 16 tables with relationships
7. **Checkpoint/Recovery** - Normal execution and crash recovery
8. **Trust Model Detail** - Enforcement points in code
9. **LLM Execution Patterns** - Sequential, pooled, batch modes
10. **Configuration Flow** - YAML → Dynaconf → Pydantic → Runtime

All diagrams use Mermaid syntax for version control compatibility.
