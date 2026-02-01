# ELSPETH C4 Architecture Diagrams

**Date:** 2026-01-27
**Analysis Lead:** Claude Opus 4.5
**Source:** Discovery findings + source code exploration

---

## Level 1: System Context

The system context diagram shows ELSPETH as a central system interacting with external actors and data systems.

```mermaid
C4Context
    title System Context Diagram: ELSPETH Auditable Pipeline Framework

    Person(operator, "Pipeline Operator", "Configures and runs data pipelines, investigates audit trails")
    Person(auditor, "Compliance Auditor", "Reviews audit exports for regulatory compliance")

    System(elspeth, "ELSPETH", "Auditable Sense/Decide/Act pipeline framework. Processes data with complete traceability.")

    System_Ext(csv_files, "CSV/JSON Files", "Source data files and output destinations")
    System_Ext(databases, "Databases", "PostgreSQL, SQLite for source/sink data")
    System_Ext(llm_apis, "LLM APIs", "Azure OpenAI, OpenAI, Anthropic for AI transforms")
    System_Ext(http_apis, "HTTP APIs", "External REST/GraphQL services")

    Rel(operator, elspeth, "Configures pipelines, monitors runs", "CLI/TUI")
    Rel(auditor, elspeth, "Exports audit data, reviews lineage", "CLI/Export")

    Rel(elspeth, csv_files, "Reads source data, writes results", "File I/O")
    Rel(elspeth, databases, "Sources/sinks data, stores audit trail", "SQLAlchemy")
    Rel(elspeth, llm_apis, "Classifies/transforms data", "HTTPS")
    Rel(elspeth, http_apis, "Enriches data", "HTTPS")
```

### Context Diagram Notes

**Key External Actors:**
- **Pipeline Operator:** Primary user who configures YAML settings, runs pipelines via CLI, and investigates issues using the audit trail
- **Compliance Auditor:** Secondary user focused on audit exports for regulatory review

**External Systems:**
- **Data Sources/Sinks:** CSV, JSON files, PostgreSQL/SQLite databases
- **AI Services:** LLM providers (Azure OpenAI primary) for classification transforms
- **HTTP APIs:** Generic REST services for data enrichment

**Critical Issue (CRIT-1):** Rate limiting for LLM/HTTP APIs is implemented but **disconnected** from the engine. The `RateLimitRegistry` exists in `core/rate_limit/` but is never instantiated by the orchestrator or executors.

---

## Level 2: Container Diagram

The container diagram shows the major subsystems (containers) within ELSPETH and their interactions.

```mermaid
C4Container
    title Container Diagram: ELSPETH Internal Architecture

    Person(operator, "Pipeline Operator", "")

    Container_Boundary(elspeth, "ELSPETH") {
        Container(cli, "CLI Layer", "Typer", "Command-line interface: run, resume, validate, explain*, plugins")
        Container(tui, "TUI Layer", "Textual", "Interactive terminal UI for explain and status (placeholder)")

        Container(engine, "Engine", "Python", "Pipeline execution: orchestration, processing, retry, executors")

        Container(landscape, "Landscape", "SQLAlchemy Core", "Audit trail: recorder, exporter, schema, lineage queries")

        Container(plugins, "Plugin System", "pluggy", "Extensible plugins: sources, transforms, gates, sinks, coalesce")

        Container(core, "Core Services", "Python", "Config, canonical JSON, DAG, events, payload store, checkpoint")
    }

    System_Ext(audit_db, "Audit Database", "SQLite/PostgreSQL")
    System_Ext(payload_fs, "Payload Store", "Filesystem")
    System_Ext(external, "External APIs", "LLM/HTTP")

    Rel(operator, cli, "elspeth run/resume/explain", "Terminal")
    Rel(cli, engine, "Orchestrates pipeline", "Python calls")
    Rel(cli, tui, "Launches TUI screens", "Textual app")

    Rel(engine, landscape, "Records audit trail", "LandscapeRecorder")
    Rel(engine, plugins, "Executes plugins", "Executor wrappers")
    Rel(engine, core, "Uses DAG, config, events", "Python imports")

    Rel(landscape, audit_db, "Stores audit data", "SQLAlchemy")
    Rel(landscape, core, "Uses canonical JSON", "Python imports")

    Rel(plugins, external, "Makes external calls", "HTTPS")
    Rel(plugins, core, "Uses payload store", "Python imports")

    Rel(core, payload_fs, "Stores large payloads", "File I/O")
```

### Container Interaction Notes

**Data Flow:**
1. **CLI** receives commands and loads config from **Core**
2. **CLI** initializes **Engine** with config and plugin instances
3. **Engine** executes plugins, recording every operation to **Landscape**
4. **Plugins** make external calls (LLM/HTTP) and return results to **Engine**
5. **Landscape** writes audit data to database and coordinates with **PayloadStore**

**Critical Issues:**
- **CRIT-4:** TUI is placeholder - `explain_screen.py` (314 LOC) and `lineage_tree.py` (198 LOC) exist but aren't wired
- **HIGH-8:** OpenTelemetry claimed in logging.py docstring but no actual tracer configuration exists

---

## Level 3: Component Diagrams

### Engine Subsystem

```mermaid
C4Component
    title Component Diagram: Engine Subsystem

    Container_Boundary(engine, "Engine (src/elspeth/engine/)") {
        Component(orchestrator, "Orchestrator", "orchestrator.py (2000 LOC)", "Full run lifecycle: init, source loading, processing, completion")
        Component(processor, "RowProcessor", "processor.py (900 LOC)", "DAG traversal with work queue, routes tokens through pipeline")

        Component(transform_exec, "TransformExecutor", "executors.py", "Wraps transform.process() with audit recording and retry")
        Component(gate_exec, "GateExecutor", "executors.py", "Evaluates routing decisions, records routing events")
        Component(agg_exec, "AggregationExecutor", "executors.py", "Stateful batching with trigger evaluation")
        Component(sink_exec, "SinkExecutor", "executors.py", "Batch writes to output sinks")
        Component(coalesce_exec, "CoalesceExecutor", "coalesce_executor.py", "Fork/join barrier with merge policies")

        Component(retry_mgr, "RetryManager", "retry.py (250 LOC)", "Tenacity-based retry with backoff (no circuit breaker)")
        Component(tokens, "TokenManager", "tokens.py", "Creates/manages token identity and lineage")
        Component(spans, "SpanFactory", "spans.py", "Creates OpenTelemetry span stubs (not connected)")
        Component(triggers, "TriggerEvaluator", "triggers.py", "Evaluates count/timeout/condition triggers")
        Component(batch_adapter, "BatchAdapter", "batch_adapter.py", "Adapts batch-aware transforms to row-level API")
    }

    Rel(orchestrator, processor, "Creates and invokes", "")
    Rel(orchestrator, tokens, "Creates source tokens", "")
    Rel(orchestrator, retry_mgr, "Passes to processor", "")

    Rel(processor, transform_exec, "Execute transforms", "")
    Rel(processor, gate_exec, "Evaluate gates", "")
    Rel(processor, agg_exec, "Accept rows into batches", "")
    Rel(processor, coalesce_exec, "Accept tokens at coalesce points", "")
    Rel(processor, sink_exec, "Write to sinks", "")

    Rel(transform_exec, retry_mgr, "Retries on failure", "")
    Rel(transform_exec, spans, "Creates execution spans", "")
    Rel(transform_exec, batch_adapter, "For batch-aware transforms", "")

    Rel(agg_exec, triggers, "Evaluates batch triggers", "")
```

#### Engine Issues Highlighted

| Component | Issue ID | Description |
|-----------|----------|-------------|
| Orchestrator | HIGH-13 | 123 lines duplicated between `_execute_pipeline()` and `_execute_pipeline_with_instances()` |
| RowProcessor | CRIT-3 | `check_timeouts()` in CoalesceExecutor never called during processing |
| RetryManager | Arch | No circuit breaker - cascading failures if external service down |
| SpanFactory | HIGH-8 | Creates spans but no tracer configured - spans go nowhere |
| CoalesceExecutor | HIGH-3 | `_completed_keys` grows unbounded - memory leak |

---

### Landscape Subsystem

```mermaid
C4Component
    title Component Diagram: Landscape Subsystem

    Container_Boundary(landscape, "Landscape (src/elspeth/core/landscape/)") {
        Component(recorder, "LandscapeRecorder", "recorder.py (2457 LOC)", "High-level API for audit recording: runs, nodes, tokens, states")
        Component(exporter, "LandscapeExporter", "exporter.py", "Exports audit data for compliance review (N+1 query pattern!)")
        Component(schema, "Schema", "schema.py", "SQLAlchemy table definitions (14 tables)")
        Component(database, "LandscapeDB", "database.py", "Connection management, in_memory factory")
        Component(repos, "Repositories", "repositories.py", "Row-to-object conversions for each table")
        Component(db_ops, "DatabaseOps", "_database_ops.py", "Reduced boilerplate for common operations")
        Component(row_data, "RowDataResult", "row_data.py", "Row state reconstruction for explain queries")
        Component(repro, "ReproducibilityGrade", "reproducibility.py", "Grades run reproducibility (deterministic/seeded/etc)")
    }

    Rel(recorder, database, "Executes queries", "SQLAlchemy")
    Rel(recorder, repos, "Converts rows to objects", "")
    Rel(recorder, db_ops, "Uses for inserts/updates", "")
    Rel(recorder, schema, "References tables", "")

    Rel(exporter, recorder, "Uses to fetch data", "")
    Rel(exporter, database, "Direct queries for export", "")

    Rel(database, schema, "Creates tables from", "metadata.create_all()")

    Rel(row_data, recorder, "Queries for row state", "")
    Rel(repro, recorder, "Queries run metadata", "")
```

#### Landscape Issues Highlighted

| Component | Issue ID | Description |
|-----------|----------|-------------|
| LandscapeRecorder | HIGH-1 | `checkpoints_table` defined in schema but no `create_checkpoint()` method |
| LandscapeRecorder | MED-3 | Call index counter in-memory only - resume could create duplicates |
| LandscapeExporter | HIGH-2 | N+1 query pattern: 21,001 queries for 1000 rows |
| Schema | MED-9 | Missing composite index on `token_outcomes(run_id, token_id)` |
| LandscapeDB | MED-8 | `in_memory()` factory bypasses schema validation |

---

### Plugin Subsystem

```mermaid
C4Component
    title Component Diagram: Plugin Subsystem

    Container_Boundary(plugins, "Plugins (src/elspeth/plugins/)") {
        Component(protocols, "Protocols", "protocols.py", "Plugin contracts: Source, Transform, Gate, Sink, Coalesce")
        Component(base, "Base Classes", "base.py", "Default implementations: BaseSource, BaseTransform, BaseGate, BaseSink")
        Component(manager, "PluginManager", "manager.py", "pluggy-based registration and discovery")
        Component(discovery, "Discovery", "discovery.py", "Auto-discovers plugins from entry points")
        Component(validation, "Validation", "validation.py", "Schema validation (hardcoded plugin lookup!)")
        Component(context, "PluginContext", "context.py", "Runtime context passed to plugins")

        Component(sources, "Sources", "sources/", "CSVSource, JSONSource, NullSource")
        Component(transforms, "Transforms", "transforms/", "FieldMapper, Passthrough, Truncate, etc.")
        Component(sinks, "Sinks", "sinks/", "CSVSink, JSONSink, DatabaseSink")
        Component(llm, "LLM Transforms", "llm/", "Azure batch, multi-query, templates")
        Component(clients, "Clients", "clients/", "HTTP, LLM, Replayer, Verifier")
    }

    Rel(manager, protocols, "Validates against", "runtime_checkable")
    Rel(manager, discovery, "Uses for auto-discovery", "")

    Rel(base, protocols, "Implements", "")
    Rel(sources, base, "Extends BaseSource", "")
    Rel(transforms, base, "Extends BaseTransform", "")
    Rel(sinks, base, "Extends BaseSink", "")

    Rel(llm, transforms, "Extends transforms", "")
    Rel(llm, clients, "Uses LLM client", "")

    Rel(validation, sources, "Hardcoded lookup!", "")
    Rel(validation, transforms, "Hardcoded lookup!", "")
```

#### Plugin Issues Highlighted

| Component | Issue ID | Description |
|-----------|----------|-------------|
| Protocols + Base | HIGH-6 | Duality creates maintenance burden - `_on_error` already has doc drift |
| Protocols | Missing | `CoalesceProtocol` exists but no `BaseCoalesce` class |
| Validation | MED-4 | Hardcoded plugin lookup tables instead of using PluginManager |
| LLM Transforms | HIGH-10 | `process()` raises NotImplementedError - violates LSP |
| LLM Transforms | CRIT-2 | Defensive `.get()` chain on Azure response violates trust model |
| HTTP Client | HIGH-4 | Silent JSON parse fallback - returns string instead of dict |

---

### Core Subsystem

```mermaid
C4Component
    title Component Diagram: Core Subsystem

    Container_Boundary(core, "Core (src/elspeth/core/)") {
        Component(config, "Configuration", "config.py (1228 LOC)", "Pydantic models + Dynaconf loading for pipeline settings")
        Component(canonical, "Canonical JSON", "canonical.py", "Two-phase RFC 8785 serialization for deterministic hashing")
        Component(dag, "ExecutionGraph", "dag.py", "NetworkX-based DAG with domain operations")
        Component(events, "EventBus", "events.py", "Synchronous pub/sub for CLI observability")
        Component(logging, "Logging", "logging.py", "structlog configuration (claims OTel - doesn't deliver)")
        Component(payload, "PayloadStore", "payload_store.py", "Content-addressable storage for large blobs")
        Component(checkpoint, "Checkpoint", "checkpoint/", "CheckpointManager for crash recovery")
        Component(retention, "Retention", "retention/", "Purge policies for payload cleanup")
        Component(security, "Security", "security/", "Secret fingerprinting via HMAC")
        Component(rate_limit, "RateLimiting", "rate_limit/", "DISCONNECTED: Registry exists but not wired to engine")
    }

    Rel(config, dag, "Triggers graph construction", "from_plugin_instances()")
    Rel(config, canonical, "Hashes config for audit", "stable_hash()")

    Rel(checkpoint, canonical, "Hashes topology", "compute_upstream_topology_hash()")
    Rel(checkpoint, dag, "Validates graph", "")

    Rel(retention, payload, "Purges old payloads", "PayloadStore.delete()")

    Rel(security, config, "Fingerprints secrets", "ELSPETH_FINGERPRINT_KEY")
```

#### Core Issues Highlighted

| Component | Issue ID | Description |
|-----------|----------|-------------|
| RateLimiting | CRIT-1 | Fully implemented but **never instantiated** by engine |
| Checkpoint | HIGH-11 | Hardcoded `cutoff_date = datetime(2026, 1, 24)` for format changes |
| PayloadStore | HIGH-7 | Duplicate protocols in `payload_store.py` and `retention/purge.py` |
| Config | Layer | Imports `ExpressionParser` from engine - layer violation |
| Logging | HIGH-8 | Docstring claims OpenTelemetry integration that doesn't exist |

---

## Data Flow Diagrams

### Pipeline Execution Flow

```mermaid
flowchart TB
    subgraph CLI["CLI Layer"]
        cmd[elspeth run]
    end

    subgraph Engine["Engine"]
        orch[Orchestrator]
        proc[RowProcessor]
        te[TransformExecutor]
        ge[GateExecutor]
        se[SinkExecutor]
        retry[RetryManager]
    end

    subgraph Plugins["Plugins"]
        source[Source]
        transform[Transform]
        gate[Gate]
        sink[Sink]
    end

    subgraph Landscape["Audit Trail"]
        recorder[LandscapeRecorder]
        db[(Audit DB)]
    end

    subgraph External["External"]
        llm[LLM API]
        http[HTTP API]
    end

    cmd --> orch
    orch --> |1. begin_run| recorder
    orch --> |2. load| source
    source --> |rows| proc

    proc --> |3. process| te
    te --> |execute| transform
    transform --> |call| llm
    transform --> |call| http
    te --> |record state| recorder
    te --> |retry on fail| retry

    proc --> |4. evaluate| ge
    ge --> |execute| gate
    ge --> |record routing| recorder

    proc --> |5. write| se
    se --> |execute| sink
    se --> |record artifact| recorder

    orch --> |6. complete_run| recorder
    recorder --> db
```

### Token Lineage Flow (Fork/Coalesce)

```mermaid
flowchart LR
    subgraph Source["Source"]
        row[row_id: R1]
    end

    subgraph Fork["Fork Gate"]
        token1[token: T1]
    end

    subgraph BranchA["Branch A"]
        tokenA[token: T1-A<br/>parent: T1]
        transformA[Transform A]
    end

    subgraph BranchB["Branch B"]
        tokenB[token: T1-B<br/>parent: T1]
        transformB[Transform B]
    end

    subgraph Coalesce["Coalesce"]
        wait{Wait for<br/>branches}
        merge[Merge Strategy]
        tokenM[token: T1-M<br/>parents: T1-A, T1-B]
    end

    subgraph Sink["Output Sink"]
        output[Final Output]
    end

    row --> token1
    token1 --> |fork| tokenA
    token1 --> |fork| tokenB
    tokenA --> transformA --> wait
    tokenB --> transformB --> wait
    wait --> |quorum met| merge --> tokenM --> output

    style wait fill:#ff9,stroke:#333
```

**Note (CRIT-3):** The wait at coalesce point can hang indefinitely during processing because `check_timeouts()` is only called during `flush_pending()` at end-of-source.

---

## Dependency Graph

### Layer Violations Detected

```mermaid
flowchart TB
    subgraph Expected["Expected Layer Order"]
        CLI_E[CLI]
        Engine_E[Engine]
        Core_E[Core]
        Contracts_E[Contracts]
        Landscape_E[Landscape]
    end

    CLI_E --> Engine_E
    Engine_E --> Core_E
    Engine_E --> Landscape_E
    Core_E --> Contracts_E
    Landscape_E --> Contracts_E

    subgraph Violations["Actual Violations"]
        contracts[contracts/results.py]
        engine_retry[engine/retry.py]
        core_config[core/config.py]
        engine_expr[engine/expression_parser.py]
        core_payload[core/payload_store.py]
        retention[core/retention/purge.py]
    end

    contracts --> |imports MaxRetriesExceeded| engine_retry
    core_config --> |imports ExpressionParser| engine_expr
    retention --> |duplicates protocol| core_payload

    style contracts fill:#f99,stroke:#333
    style core_config fill:#f99,stroke:#333
    style retention fill:#f99,stroke:#333
```

### Module Import Structure

```mermaid
graph LR
    subgraph CLI["cli.py"]
        cli_main[main]
    end

    subgraph Engine["engine/"]
        orch[orchestrator]
        proc[processor]
        exec[executors]
        retry[retry]
        tokens[tokens]
        coalesce[coalesce_executor]
    end

    subgraph Core["core/"]
        config[config]
        dag[dag]
        events[events]
        canonical[canonical]
        checkpoint[checkpoint/]
        rate_limit[rate_limit/]
    end

    subgraph Landscape["core/landscape/"]
        recorder[recorder]
        exporter[exporter]
        schema[schema]
        database[database]
    end

    subgraph Plugins["plugins/"]
        protocols[protocols]
        base[base]
        manager[manager]
        sources[sources/]
        transforms[transforms/]
        sinks[sinks/]
    end

    cli_main --> orch
    cli_main --> config

    orch --> proc
    orch --> recorder
    orch --> dag

    proc --> exec
    proc --> retry
    proc --> tokens
    proc --> coalesce

    exec --> recorder
    exec --> protocols

    recorder --> schema
    recorder --> database

    manager --> protocols
    sources --> base
    transforms --> base
    sinks --> base

    %% Violations shown in red
    config --> |VIOLATION| exec

    style config fill:#fdd
```

---

## Problematic Areas Summary

### Disconnected Code (High Risk)

```mermaid
flowchart LR
    subgraph Implemented["Fully Implemented"]
        rl_reg[RateLimitRegistry<br/>250 LOC]
        rl_limiter[RateLimiter<br/>pyrate-limiter]
        explain_screen[ExplainScreen<br/>314 LOC]
        lineage_tree[LineageTree<br/>198 LOC]
        checkpoints_table[checkpoints_table<br/>schema defined]
    end

    subgraph NotWired["NOT Wired"]
        engine[Engine]
        tui_app[TUI App]
        recorder[Recorder]
    end

    rl_reg -.-> |CRIT-1: not used| engine
    explain_screen -.-> |CRIT-4: placeholder| tui_app
    lineage_tree -.-> |CRIT-4: placeholder| tui_app
    checkpoints_table -.-> |HIGH-1: no methods| recorder

    style rl_reg fill:#f99
    style explain_screen fill:#f99
    style lineage_tree fill:#f99
    style checkpoints_table fill:#f99
```

### Performance Bottlenecks

```mermaid
flowchart TD
    subgraph Exporter["LandscapeExporter.export_run()"]
        q1[Query: get_rows]
        loop1{For each row}
        q2[Query: get_tokens]
        loop2{For each token}
        q3[Query: get_node_states]
        loop3{For each state}
        q4[Query: get_calls]
    end

    q1 --> loop1
    loop1 --> q2
    q2 --> loop2
    loop2 --> q3
    q3 --> loop3
    loop3 --> q4

    result[1000 rows = 21,001 queries<br/>HIGH-2]

    q4 --> result

    style result fill:#f99
```

---

## Confidence Assessment

**Confidence Level:** High

**Evidence Trail:**
- Read all key source files (orchestrator, processor, executors, recorder, exporter, config, dag)
- Verified disconnections via grep searches (rate limit, check_timeouts)
- Traced import graphs for layer violations
- Counted query patterns in nested loops
- Cross-referenced discovery findings with source exploration

**Diagrams Based On:**
- `01-discovery-findings.md` - 47 issues identified by 17 agents
- Direct source code reading of 20+ files
- Module structure from glob patterns
- Import analysis from file headers

**Information Gaps:**
- Did not diagram Alembic migration flow
- TUI internal state machine not fully mapped
- Batch transform internal architecture simplified
