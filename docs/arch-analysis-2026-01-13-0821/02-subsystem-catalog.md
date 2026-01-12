# Subsystem Catalog (Submodule Classification)

This is an analysis-oriented decomposition of `src/elspeth/` into cohesive submodules, with responsibilities and dependency notes to guide deeper follow-on analysis.

## 1) `elspeth.core` — Core Infrastructure

### 1.1 `elspeth.core.canonical` (Canonicalization + Stable Hashing)
- **Location:** `src/elspeth/core/canonical.py`
- **Responsibility:** Deterministic JSON serialization and stable SHA-256 hashing for audit integrity (strictly rejects NaN/Inf).
- **Key API:** `canonical_json(obj) -> str`, `stable_hash(obj) -> str`, `CANONICAL_VERSION`
- **Inbound deps:** `core.landscape.recorder`, `engine.executors`
- **Outbound deps:** `numpy`, `pandas`, `rfc8785`
- **Notes for further analysis:** Verify normalization rules cover all expected row payload types (especially datetimes, bytes, Decimals).
- **Confidence:** High (small surface; strong tests expected in `tests/core/test_canonical.py`)

### 1.2 `elspeth.core.config` (Configuration Schema + Loader)
- **Location:** `src/elspeth/core/config.py`
- **Responsibility:** Immutable settings models (Pydantic) and config loading (Dynaconf) with explicit missing-file rejection.
- **Key API:** `load_settings(config_path) -> ElspethSettings`
- **Inbound deps:** (Not currently wired into engine runtime)
- **Outbound deps:** `dynaconf`, `pydantic`
- **Notes for further analysis:** Decide how `ElspethSettings` relates to runtime `PipelineConfig` (engine uses `PipelineConfig.config: dict` today).
- **Confidence:** Medium-High (clear semantics; integration gap is a product/design question)

### 1.3 `elspeth.core.dag` (Execution Graph Primitives)
- **Location:** `src/elspeth/core/dag.py`
- **Responsibility:** Validate + query an execution DAG using NetworkX (`ExecutionGraph`, `NodeInfo`).
- **Key API:** `ExecutionGraph.add_node/add_edge/validate/topological_order/get_source/get_sinks`
- **Inbound deps:** (Not currently used by engine)
- **Outbound deps:** `networkx`
- **Notes for further analysis:** Engine currently executes linear pipelines; DAG compilation/execution looks planned but not integrated.
- **Confidence:** Medium (API is present; execution integration is pending)

### 1.4 `elspeth.core.payload_store` (Content-Addressable Blob Storage)
- **Location:** `src/elspeth/core/payload_store.py`
- **Responsibility:** Protocol + filesystem implementation for storing large payloads by hash.
- **Key API:** `PayloadStore` protocol, `FilesystemPayloadStore`
- **Inbound deps:** `core.landscape.recorder` (optional retrieval), future engine ingestion
- **Outbound deps:** `pathlib`, `hashlib`
- **Notes for further analysis:** Current engine path doesn’t store payload refs for source rows; only retrieval exists in LandscapeRecorder.
- **Confidence:** Medium-High

### 1.5 `elspeth.core.landscape` (Audit Backbone)
- **Location:** `src/elspeth/core/landscape/`
- **Responsibility:** Persist the full audit trail for runs, nodes, edges, tokens, states, routing, calls, artifacts, and batches.
- **Key components:**
  - `schema.py` — SQLAlchemy Core tables + indexes
  - `models.py` — dataclasses mirroring table records
  - `database.py` — engine creation + SQLite pragmas + transaction context manager
  - `recorder.py` — high-level API to create/query audit records
- **Inbound deps:** `engine.*` uses `LandscapeRecorder` heavily (node states, routing events, batches, artifacts).
- **Outbound deps:** `sqlalchemy`, `core.canonical`; **also depends on** `plugins.enums` (`NodeType`, `Determinism`).
- **Notes for further analysis:**
  - Layering: core importing `plugins.enums` may be intentional (“shared types”) but violates a strict core→plugins layering.
  - Audit invariants are enforced at runtime (e.g., `MissingEdgeError` from engine when routing edge missing).
- **Confidence:** High (largest but well-scoped; extensive tests under `tests/core/landscape/`)

## 2) `elspeth.engine` — SDA Execution Engine

### 2.1 `elspeth.engine.orchestrator` (Run Lifecycle + Wiring)
- **Location:** `src/elspeth/engine/orchestrator.py`
- **Responsibility:** Creates a run, registers nodes/edges, iterates source rows, delegates per-row processing, writes sinks, completes run.
- **Key types:** `PipelineConfig`, `RunResult`, `Orchestrator`
- **Inbound deps:** Expected to be called by CLI/app layer (not currently present).
- **Outbound deps:** `core.landscape`, `engine.processor`, `engine.spans`, `engine.executors`, `plugins.context`
- **Notes for further analysis:** Orchestrator sets `node_id` onto plugin instances dynamically (protocols don’t include `node_id`).
- **Confidence:** Medium (works for linear runs; DAG/fork/coalesce paths are partial)

### 2.2 `elspeth.engine.processor` (Row-by-Row Pipeline Execution)
- **Location:** `src/elspeth/engine/processor.py`
- **Responsibility:** Creates initial tokens, runs transforms/gates/aggregations in sequence, returns `RowResult` with terminal-ish outcome.
- **Key types:** `RowProcessor`, `RowResult`
- **Inbound deps:** Orchestrator
- **Outbound deps:** `engine.executors`, `engine.tokens`
- **Notes for further analysis:** Implementation explicitly notes “linear pipelines only”; fork handling returns `"forked"` without scheduling child tokens.
- **Confidence:** Medium

### 2.3 `elspeth.engine.executors` (Audit-Wrapper Adapters Around Plugin Calls)
- **Location:** `src/elspeth/engine/executors.py`
- **Responsibility:** Wrap plugin calls with timing, spans, Landscape node state recording, routing events, batch state transitions, artifact registration.
- **Key types:** `TransformExecutor`, `GateExecutor`, `AggregationExecutor`, `SinkExecutor`, `MissingEdgeError`
- **Inbound deps:** `RowProcessor`, `Orchestrator`
- **Outbound deps:** `core.landscape`, `core.canonical`, `engine.spans`, `engine.tokens`, `plugins.results`
- **Notes for further analysis:**
  - Gate routing requires pre-registered edges via `(node_id, label) -> edge_id` map; missing edges are treated as audit integrity violations.
  - Sink execution currently assumes a “batch sink” interface (`write(rows)->artifact info`), distinct from `plugins.protocols.SinkProtocol`.
- **Confidence:** Medium-High (clear invariants; check integration gaps for sink adapter + retries)

### 2.4 `elspeth.engine.tokens` (Token Identity + Fork/Coalesce Helpers)
- **Location:** `src/elspeth/engine/tokens.py`
- **Responsibility:** Token identity wrapper (`TokenInfo`) and token lifecycle ops via LandscapeRecorder.
- **Key types:** `TokenInfo`, `TokenManager`
- **Inbound deps:** `RowProcessor`, `GateExecutor`
- **Outbound deps:** `core.landscape.recorder`
- **Confidence:** High

### 2.5 `elspeth.engine.spans` (Tracing Facade)
- **Location:** `src/elspeth/engine/spans.py`
- **Responsibility:** Provide consistent span contexts; no-op when tracer absent.
- **Key type:** `SpanFactory`
- **Outbound deps:** Optional `opentelemetry` (type-check only)
- **Confidence:** High

### 2.6 `elspeth.engine.retry` (Retry Policy + tenacity Wrapper)
- **Location:** `src/elspeth/engine/retry.py`
- **Responsibility:** Centralize retry behavior; exposes `RetryConfig` and `RetryManager.execute_with_retry`.
- **Outbound deps:** `tenacity`
- **Notes for further analysis:** Not currently wired into `TransformExecutor` usage; it’s an available building block.
- **Confidence:** Medium-High

## 3) `elspeth.plugins` — Plugin Contracts + Registration

### 3.1 `elspeth.plugins.protocols` (Typed Contracts)
- **Location:** `src/elspeth/plugins/protocols.py`
- **Responsibility:** Define `SourceProtocol`, `TransformProtocol`, `GateProtocol`, `AggregationProtocol`, `CoalesceProtocol`, `SinkProtocol`.
- **Inbound deps:** `plugins.base` conceptually implements these; engine often uses `Any` and duck-typing (`hasattr`).
- **Confidence:** High (contracts are explicit; runtime enforcement is external)

### 3.2 `elspeth.plugins.schemas` (Schema + Compatibility Checking)
- **Location:** `src/elspeth/plugins/schemas.py`
- **Responsibility:** Pydantic-based schema base + row validation + producer/consumer compatibility checks.
- **Key API:** `validate_row`, `check_compatibility`
- **Confidence:** Medium-High (compatibility is intentionally conservative; follow-on analysis should compare vs actual pipeline needs)

### 3.3 `elspeth.plugins.results` (Engine↔Plugin Result Contracts)
- **Location:** `src/elspeth/plugins/results.py`
- **Responsibility:** Result objects (`TransformResult`, `GateResult`, `AcceptResult`) and routing action (`RoutingAction`) with immutability constraints.
- **Confidence:** High

### 3.4 `elspeth.plugins.context` (Plugin Runtime Context)
- **Location:** `src/elspeth/plugins/context.py`
- **Responsibility:** Carries run/config metadata and optional integrations (landscape, tracer, payload_store).
- **Key API:** `PluginContext.get()`, `PluginContext.start_span()`
- **Notes for further analysis:** The `landscape` attribute is typed as a minimal protocol; engine currently passes the full LandscapeRecorder (type ignored).
- **Confidence:** Medium

### 3.5 `elspeth.plugins.manager` + `hookspecs` (pluggy Registration + Discovery)
- **Location:** `src/elspeth/plugins/manager.py`, `src/elspeth/plugins/hookspecs.py`
- **Responsibility:** Register plugins via pluggy hooks; maintain name→class caches; detect duplicates.
- **Inbound deps:** Future CLI/app would likely load plugin packs and use manager to instantiate pipeline plugins.
- **Notes for further analysis:** Engine currently bypasses PluginManager and works with concrete instances directly.
- **Confidence:** Medium-High

## Cross-Cutting Concerns (useful for targeted follow-on analysis)
- **Audit integrity:** `stable_hash` + Landscape invariants enforced by engine executor behavior (e.g., `MissingEdgeError`).
- **Layering tension:** Types like `NodeType` and `Determinism` are needed in core+engine+plugins; currently located in `plugins.enums` but imported by core.
- **Phase alignment:** Several interfaces are “future-facing” (CLI, sink adapter, DAG execution); classify these as integration seams rather than bugs.

