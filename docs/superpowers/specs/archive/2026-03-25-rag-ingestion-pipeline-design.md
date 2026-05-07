# RAG Ingestion Pipeline — Design Spec

**Date:** 2026-03-25
**Status:** Draft (R3 — specialist review fixes: ExpressionParser replaces eval(), bootstrap_and_run extraction, unified CollectionReadinessResult, explicit collection_probes config, depth limit, indexed_at timestamp, expanded testing strategy with audit trail verification)
**Target ChromaDB version:** `chromadb >= 0.4`
**Scope:** ChromaSink plugin, pipeline `depends_on` mechanism, commencement gates, RAG retrieval readiness contract

## Overview

Complete the RAG story in ELSPETH by adding the ingestion side: a ChromaDB sink plugin that writes documents into a vector store, a `depends_on` mechanism that sequences pipeline runs, commencement gates that evaluate go/no-go conditions before a pipeline starts, and a readiness contract on the RAG retrieval transform that refuses to run against an empty collection.

This is the dogfood for three new engine capabilities. The ChromaSink is the first vector store sink. The `depends_on` and commencement gate mechanisms are the first pipeline-level orchestration primitives. The readiness contract is the first plugin pre-condition check.

## Motivation

ELSPETH's RAG retrieval transform queries a pre-populated vector store, but the framework has no way to populate that store. The current workaround is a standalone `seed_collection.py` script that runs outside ELSPETH's audit model. This creates an audit gap: an auditor asking "what reference documents were available when row 42 was classified?" gets no answer from the Landscape, because the seeding happened off-book.

Bringing ingestion into ELSPETH as a proper pipeline closes this gap. Every document indexed into the vector store gets its own `token_id`, audit trail entry, and canonical content hash. The `depends_on` mechanism ensures the indexing pipeline runs before the query pipeline. Commencement gates verify the corpus is ready. The readiness contract on the RAG transform is a belt-and-suspenders guard.

### Design Principles

- **Indexing and querying are separate pipelines.** They have different schedules, failure modes, and scaling profiles. Forcing them into one pipeline creates coupling that doesn't need to exist.
- **The plugin doesn't know how the data got there.** The RAG transform's readiness check verifies the collection has data. It doesn't care whether that data came from a `depends_on` pipeline, a manual script, or an external indexer.
- **Fail loudly at the earliest possible moment.** A pipeline that would produce a run full of zero-chunk retrievals should never start.

## Implementation Staging

The design is implemented across five sub-plans. Sub-plan 1 is the critical path — it unblocks sub-plans 2, 3, and 4, which can then proceed in parallel. Sub-plan 5 assembles the end-to-end example after the others merge.

```
Sub-plan 1: Shared Infrastructure
    │
    ├──► Sub-plan 2: ChromaSink          ──┐
    ├──► Sub-plan 3: depends_on + Gates  ──┼──► Sub-plan 5: End-to-End Example
    └──► Sub-plan 4: Readiness Contract  ──┘
```

### Sub-plan 1: Shared Infrastructure

Foundation types and utilities that all other sub-plans depend on.

| Deliverable | Layer | Description |
|-------------|-------|-------------|
| `CollectionReadinessResult` | L0 (`contracts/probes.py`) | Unified frozen dataclass for collection readiness — used by probes, providers, and transforms |
| `CollectionProbe` protocol | L0 (`contracts/probes.py`) | Protocol for collection readiness probes |
| `ChromaConnectionConfig` | L3 (`plugins/infrastructure/`) | Shared Pydantic model for ChromaDB connection fields — used by sink, provider, and probe configs |
| `ExpressionParser` extension | L1 (`core/expression_parser.py`) | Extend existing AST-whitelist parser to accept plain dict contexts (currently only `PipelineRow`) |
| Error types | L0 (`contracts/errors.py`) | `DependencyFailedError`, `CommencementGateFailedError`, `RetrievalNotReadyError` |

**Risk:** Low. Small, self-contained changes. No engine modifications.
**Review focus:** L0 type design, freeze guard decisions, `ExpressionParser` dict-context extension.

### Sub-plan 2: ChromaSink Plugin

New sink plugin that writes rows into ChromaDB collections.

| Deliverable | Layer | Description |
|-------------|-------|-------------|
| `ChromaSinkConfig` | L3 (`plugins/sinks/`) | Pydantic config model composing `ChromaConnectionConfig` + field mapping |
| `ChromaSink` | L3 (`plugins/sinks/chroma_sink.py`) | Sink implementation — lifecycle, write, audit recording |
| Unit tests | — | Config validation, write modes, audit recording, `AuditIntegrityError` path |
| Integration test | — | CSV → ChromaSink with real ephemeral ChromaDB, Landscape assertions |

**Depends on:** Sub-plan 1 (`ChromaConnectionConfig`, `CollectionReadinessResult`).
**Risk:** Medium. New plugin, but follows established sink patterns (database_sink.py as reference).
**Review focus:** Tier model compliance, `allow_coercion=False`, content hashing before write, audit integrity.

### Sub-plan 3: `depends_on` + Commencement Gates

Engine-level orchestration primitives — the highest-complexity sub-plan.

| Deliverable | Layer | Description |
|-------------|-------|-------------|
| `bootstrap_and_run()` | L2 (`engine/`) | Extract reusable pipeline bootstrap from CLI codepath |
| `DependencyConfig`, `DependencyRunResult` | L1 (`core/config.py`) | Config model and result dataclass with `indexed_at` |
| `CommencementGateConfig`, `GateResult` | L1 (`core/config.py`) | Config model and result dataclass with frozen `context_snapshot` |
| `CollectionProbeConfig` | L1 (`core/config.py`) | Config model for explicit probe declarations |
| Dependency resolution phase | L2 (`engine/orchestrator/`) | Sequential execution, cycle detection (DFS, `Path.resolve()`), 3-level depth limit |
| Commencement gate phase | L2 (`engine/orchestrator/`) | Pre-flight context assembly, `ExpressionParser` evaluation, TOCTOU-safe freezing |
| Collection probe assembly | L3 (`plugins/infrastructure/`) | `build_collection_probes()` from explicit config declarations |
| Unit tests | — | Cycle detection (1/2/3-hop), depth limit, gate AST validation, Hypothesis property test, signal handling |
| Integration tests | — | Two-pipeline dependency, gate evaluation, nested ordering, resume behaviour |

**Depends on:** Sub-plan 1 (`CollectionReadinessResult`, `ExpressionParser`, error types).
**Risk:** High. Modifies the orchestrator's `run()` method. `bootstrap_and_run()` extraction touches the CLI codepath. Needs careful review.
**Review focus:** `bootstrap_and_run()` equivalence with CLI path, cycle detection correctness, TOCTOU closure, `KeyboardInterrupt` propagation.

### Sub-plan 4: Readiness Contract

Pre-condition check on the existing RAG retrieval transform.

| Deliverable | Layer | Description |
|-------------|-------|-------------|
| `check_readiness()` on `RetrievalProvider` | L3 (`plugins/infrastructure/`) | Protocol extension returning `CollectionReadinessResult` |
| `ChromaSearchProvider.check_readiness()` | L3 | Collection existence + count check |
| `AzureSearchProvider.check_readiness()` | L3 | Index existence + document count via HTTP |
| `RAGRetrievalTransform.on_start()` guard | L3 (`plugins/transforms/rag/`) | Readiness check after provider construction |
| Unit tests | — | Ready/empty/missing/unreachable states |
| Integration test | — | Transform against empty vs. populated collection |

**Depends on:** Sub-plan 1 (`CollectionReadinessResult`).
**Risk:** Medium-low. Small scope but modifies a `@runtime_checkable` Protocol — all existing test doubles and mocks of `RetrievalProvider` must be updated in the same commit.
**Review focus:** Protocol evolution, `spec_set` on mocks, single-attempt semantics.

### Sub-plan 5: End-to-End Example + Smoke Tests

Final integration stage after all four sub-plans merge.

| Deliverable | Description |
|-------------|-------------|
| `examples/chroma_rag_indexed/` | Indexing pipeline + query pipeline with `depends_on`, gates, and retrieval |
| `documents.csv` | Sample reference corpus |
| `questions.csv` | Sample questions |
| Integration smoke test | Full pipeline sequence: dependency → gate → readiness → retrieval |

**Depends on:** Sub-plans 2, 3, 4 (all merged).
**Risk:** Low. Assembly and verification only — no new code beyond example configs and the smoke test.

## Architecture

### File Layout

New files:

```
src/elspeth/plugins/
├── sinks/
│   └── chroma_sink.py                          # ChromaSink plugin
└── infrastructure/
    └── clients/
        └── retrieval/
            ├── base.py                          # +check_readiness() on protocol (returns CollectionReadinessResult from L0)
            ├── types.py                         # (no new types — ReadinessResult unified into L0)
            ├── chroma.py                        # +check_readiness() implementation
            └── azure_search.py                  # +check_readiness() implementation

src/elspeth/core/
└── config.py                                    # +DependencyConfig, +CommencementGateConfig

src/elspeth/engine/
└── orchestrator/
    └── core.py                                  # +dependency resolution phase
                                                 # +commencement gate evaluation phase

src/elspeth/contracts/
├── probes.py                                    # +CollectionProbe protocol, +CollectionReadinessResult
└── errors.py                                    # +DependencyFailedError
                                                 # +CommencementGateFailedError
                                                 # +RetrievalNotReadyError
```

Modified files:

```
src/elspeth/plugins/transforms/rag/transform.py  # +readiness check in on_start()
src/elspeth/plugins/infrastructure/discovery.py   # No change needed (auto-discovers sinks/)
```

### Layer Placement

All new code lives in existing layers. One cross-layer dependency requires a protocol-based injection pattern (see Commencement Gates, Collection Probing):

| Component | Layer | Rationale |
|-----------|-------|-----------|
| `ChromaSink` | L3 (plugins/sinks) | Sink plugin, same as CSV/JSON/Database sinks |
| `CollectionReadinessResult` | L0 (contracts/probes) | Unified result type for all collection readiness checks |
| `check_readiness()` | L3 (plugins/infrastructure) | Provider method, alongside `search()` |
| `CollectionProbe` protocol | L0 (contracts/) | Protocol for collection readiness probes, defined in contracts so L2 can depend on it |
| `DependencyConfig` | L1 (core/config) | Pipeline config, alongside `RetrySettings` |
| `CommencementGateConfig` | L1 (core/config) | Pipeline config, alongside `RetrySettings` |
| `DependencyRunResult` | L1 (core/config) | Frozen dataclass for dependency run metadata |
| `GateResult` | L1 (core/config) | Frozen dataclass for gate evaluation metadata |
| Dependency resolution | L2 (engine/orchestrator) | Engine orchestration logic |
| Commencement gate evaluation | L2 (engine/orchestrator) | Engine orchestration logic, receives probes via injection |
| Collection probe implementations | L3 (plugins/infrastructure) | ChromaDB/Azure implementations of `CollectionProbe` |
| Error types | L0 (contracts/errors) | Shared error types |

## Component 1: ChromaSink Plugin

### Purpose

Write pipeline rows into a ChromaDB collection. Each row becomes a document. ChromaDB handles embedding internally via its configured embedding function.

### Config Model

```yaml
sinks:
  corpus_output:
    plugin: chroma_sink
    options:
      collection: science-facts
      mode: persistent                  # persistent | client
      persist_directory: ./chroma_data  # Required for persistent mode
      # OR for client mode:
      # host: chroma.internal
      # port: 8000
      distance_function: cosine         # cosine | l2 | ip

      field_mapping:
        document: text_content          # Row field -> Chroma document text
        id: doc_id                      # Row field -> Chroma document ID
        metadata:                       # Row fields -> Chroma metadata dict
          - topic
          - subtopic

      on_duplicate: overwrite           # overwrite | skip | error

      schema:
        mode: fixed
        fields:
          - "doc_id: str"
          - "text_content: str"
          - "topic: str"
          - "subtopic: str"

landscape:
  url: sqlite:///./runs/audit.db
```

### Config Class: `ChromaSinkConfig`

Extends `DataPluginConfig`. Pydantic model with:

- **`collection`** (str, required): ChromaDB collection name.
- **`mode`** (Literal["persistent", "client"], required): How to connect. Mirrors `ChromaSearchProviderConfig`.
- **`persist_directory`** (str, required if mode=persistent): Path to ChromaDB data directory. Resolved relative to the pipeline config file.
- **`host`** (str, required if mode=client): ChromaDB server hostname.
- **`port`** (int, default 8000, required if mode=client): ChromaDB server port.
- **`distance_function`** (Literal["cosine", "l2", "ip"], default "cosine"): Passed to `get_or_create_collection()` as `hnsw:space` metadata.
- **`field_mapping`** (FieldMappingConfig, required): Maps row fields to Chroma concepts. Sub-model with:
  - `document` (str): Row field containing text to embed.
  - `id` (str): Row field containing document ID.
  - `metadata` (list[str]): Row fields to include as Chroma metadata.
- **`on_duplicate`** (Literal["overwrite", "skip", "error"], default "overwrite"): Behaviour when a document ID already exists.

Field mapping is required, not optional. No convention-based defaults.

**Validation rules:**
- `persist_directory` required when `mode=persistent`, forbidden when `mode=client`.
- `host` required when `mode=client`, forbidden when `mode=persistent`.
- All `field_mapping` fields must appear in the schema's field list.
- `field_mapping.document` and `field_mapping.id` must be `str` type fields.
- `field_mapping.metadata` fields must be `str`, `int`, `float`, or `bool` type fields in the schema. This matches ChromaDB's metadata type constraint and ensures canonical JSON hashing works without type-specific serialization surprises.
- Client mode: HTTPS required unless host is localhost (mirrors `ChromaSearchProviderConfig`).

### Lifecycle

**`__init__(config: dict)`:**
- Parse `ChromaSinkConfig.from_dict(config)`.
- Build input schema via `create_schema_from_config(..., allow_coercion=False)`. Sinks are Tier 2 — wrong types are an upstream bug.

**`on_start(ctx: LifecycleContext)`:**
- Construct ChromaDB client (PersistentClient or HttpClient depending on mode).
- Verify connection: `client.heartbeat()` for client mode, collection access for persistent mode.
- Get or create collection with configured `distance_function`.
- Capture `run_id`, `landscape`, `telemetry_emit` from context.
- If connection fails: raise — pipeline never starts.

**`write(rows: list[dict], ctx: SinkContext) -> ArtifactDescriptor`:**
- Extract mapped fields from each row using `field_mapping`.
- Compute canonical JSON hash of the payload before writing (proves intent).
- Execute write operation based on `on_duplicate`:
  - `overwrite`: `collection.upsert(ids, documents, metadatas)`
  - `skip`: Pre-filter via `collection.get(ids)` to identify existing IDs, then `collection.add()` only the new ones. Skipped IDs are recorded in the call audit (documents we chose not to write).
  - `error`: Pre-check via `collection.get(ids)`. If any IDs exist, raise a pipeline-level error with the duplicate IDs listed. The batch is not written. This is a pipeline-level failure, not a row-level quarantine, because ChromaDB batch operations are all-or-nothing.
- Record call via `ctx.record_call(call_type=EXTERNAL_API, provider="chromadb", ...)`.
- Return `ArtifactDescriptor` with content hash, payload size, row count.
- If audit recording fails after a successful write: raise `AuditIntegrityError` — data was written but not recorded.

**`flush()`:**
- No-op — ChromaDB writes are committed atomically by the server/engine. No explicit flush required from the client side.

**`on_complete(ctx: LifecycleContext)`:**
- Emit telemetry: total documents written, total bytes, collection name.

**`close()`:**
- Release client reference.

### Audit Trail

Every `write()` batch records via `ctx.record_call()`:

| Field | Value |
|-------|-------|
| `call_type` | `CallType.EXTERNAL_API` |
| `provider` | `"chromadb"` |
| `status` | `CallStatus.SUCCESS` or `CallStatus.ERROR` |
| `request_data` | `{operation, collection, row_count, document_ids}` |
| `response_data` | `{rows_written}` |
| `latency_ms` | Wall-clock time of the upsert/add call |

Content hash is computed from the Canonical subsystem's two-phase canonicalization of the payload before the write. This proves what was sent, regardless of what ChromaDB does internally (embedding, tokenization).

### Error Handling

| Scenario | Response |
|----------|----------|
| ChromaDB unreachable in `on_start()` | Raise — pipeline never starts |
| ChromaDB unreachable during `write()` | Raise — engine retry policy applies |
| Row missing a mapped field | Crash — Tier 2 violation, upstream bug |
| Duplicate ID with `on_duplicate: error` | Pipeline-level error — pre-check IDs via `collection.get(ids)` before write, raise if any exist. Since ChromaDB's `add()`/`upsert()` are batch operations, individual row quarantine is not possible. The entire batch fails. |
| Audit recording fails after successful write | Raise `AuditIntegrityError` — data written but unrecorded |

### Config Symmetry with Retrieval Provider — Shared `ChromaConnectionConfig`

Extract the connection fields into a shared Pydantic model to eliminate validation duplication:

```python
class ChromaConnectionConfig(BaseModel):
    """Shared ChromaDB connection config — used by ChromaSinkConfig,
    ChromaSearchProviderConfig, and CollectionProbeConfig."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    collection: str
    mode: Literal["persistent", "client"]
    persist_directory: str | None = None
    host: str | None = None
    port: int = 8000
    distance_function: Literal["cosine", "l2", "ip"] = "cosine"
```

`ChromaSinkConfig` and `ChromaSearchProviderConfig` compose this model (adding their own fields like `field_mapping` and `on_duplicate`). `CollectionProbeConfig.provider_config` also uses it. Validation rules (mode/host/persist_directory mutual exclusion, HTTPS enforcement) live in one place.

When an indexing pipeline and a query pipeline target the same store, the connection config is visually identical and mechanically identical. Misconfiguration between the two is caught by the shared validation.

## Component 2: Pipeline `depends_on` Mechanism

### Purpose

Declare that a pipeline requires other pipelines to run first. Dependencies are fully independent pipeline runs with their own `run_id`, Landscape records, and checkpoint streams.

### Config Shape

New top-level key in pipeline YAML:

```yaml
depends_on:
  - name: index_corpus              # Unique label within the list
    settings: ./index_pipeline.yaml # Path relative to parent config file
```

- `name` (str, required): Human-readable label. Must be unique within the `depends_on` list.
- `settings` (str, required): Path to the dependency pipeline's settings file. Resolved relative to the parent pipeline's settings file directory.

### Config Model: `DependencyConfig`

Pydantic model added to `core/config.py`:

```python
class DependencyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    settings: str
```

`ElspethSettings` gains an optional field:

```python
depends_on: list[DependencyConfig] = Field(default_factory=list)
```

### Execution Semantics

1. Dependencies run sequentially in declared order, before anything else in the main pipeline.
2. For each dependency, the orchestrator:
   a. Resolves the `settings` path relative to the parent config file.
   b. Loads the dependency's `ElspethSettings`.
   c. Creates a new `Orchestrator` instance.
   d. Calls `run()` on the child orchestrator.
   e. Records the result (run_id, status, duration).
3. If any dependency run fails (any non-success terminal state), the main pipeline aborts with `DependencyFailedError`. The error includes the dependency name, its `run_id`, and the failure reason.
4. If all dependencies succeed, execution continues to commencement gates (Component 3), then the main pipeline.

### Child Orchestrator Independence

The child orchestrator is fully independent:

- Own `run_id` — not a child of the parent's run_id.
- Own Landscape records — written to whichever database the dependency's config specifies (which may be the same database as the parent — recommended for audit correlation).
- Own checkpoint stream — the dependency can be resumed independently if it was interrupted in a previous manual run.
- No shared plugin instances, rate limiters, or telemetry sinks.

This means dependency pipelines are testable in isolation: `elspeth run --settings index_pipeline.yaml --execute` produces identical behaviour to when it runs via `depends_on`.

### Landscape Correlation

When the main pipeline's run record is created, it includes a `dependency_runs` metadata field:

```python
{
    "dependency_runs": [
        {
            "name": "index_corpus",
            "run_id": "abc-123",
            "settings_hash": "sha256:...",
            "duration_ms": 4520,
            "indexed_at": "2026-03-25T14:02:33Z"
        }
    ]
}
```

- `settings_hash`: SHA-256 via the Canonical subsystem's two-phase canonicalization of the dependency's config. Enables drift detection — if the indexing pipeline config changes between runs, the hash changes.
- Stored in the existing run metadata mechanism (freeform JSON blob on the run record) — no Landscape schema migration needed.

### `DependencyRunResult`

Frozen dataclass in `core/config.py`:

```python
@dataclass(frozen=True, slots=True)
class DependencyRunResult:
    name: str
    run_id: str
    settings_hash: str
    duration_ms: int
    indexed_at: str  # ISO 8601 timestamp of dependency completion
```

All fields are scalars — no `__post_init__` freeze guard needed.

The `status` field was removed — the existence of a `DependencyRunResult` already implies successful completion (failed dependencies raise `DependencyFailedError` before construction). When deserializing from the Landscape, validate that the record exists and crash if the expected structure is missing (Tier 1 read guard).

The `indexed_at` field records when the dependency completed. This prevents the "Drifting Goals" dynamic where corpus freshness erodes to mere presence — operators and future `if_stale` conditions can reason about when the data was last indexed, not just that data exists.

### Engine Changes

The orchestrator's `run()` method gains a new phase inserted before database initialization:

```
[NEW]      Dependency resolution phase
[NEW]      Commencement gate evaluation phase (Component 3)
[EXISTING] Database initialization phase (run record now includes dependency metadata)
[EXISTING] Plugin instantiation and on_start() phase
[EXISTING] Row processing phase
[EXISTING] Cleanup phase
```

The dependency resolution phase:

**Pre-requisite:** Extract the current CLI/entry-point pipeline bootstrap sequence into a reusable function:

```python
def bootstrap_and_run(settings_path: Path) -> RunResult:
    """Full pipeline bootstrap and execution from a settings file path.

    Handles the complete lifecycle: config loading, plugin instantiation,
    ExecutionGraph construction, LandscapeDB setup, PayloadStore creation,
    and orchestrator execution. Returns a RunResult with run_id, status,
    duration, and failure reason (if any).

    This function is used by both the CLI `execute` command and the
    dependency resolution phase. It ensures dependency pipelines go
    through the identical code path as manually-invoked pipelines.
    """
```

This extraction is a pre-requisite for the `depends_on` implementation. The current `Orchestrator.__init__` requires `LandscapeDB`, and `run()` requires `PipelineConfig`, `ExecutionGraph`, `PayloadStore`, etc. — significant construction complexity that must not be duplicated in `_resolve_dependencies`.

```python
def _resolve_dependencies(self, settings: ElspethSettings) -> list[DependencyRunResult]:
    results = []
    for dep in settings.depends_on:
        dep_settings_path = self._resolve_relative_path(dep.settings)

        run_result = bootstrap_and_run(dep_settings_path)

        if not run_result.success:
            raise DependencyFailedError(
                dependency_name=dep.name,
                run_id=run_result.run_id,
                reason=run_result.failure_reason,
            )

        results.append(DependencyRunResult(
            name=dep.name,
            run_id=run_result.run_id,
            settings_hash=hash_settings_file(dep_settings_path),
            duration_ms=run_result.duration_ms,
            indexed_at=run_result.completed_at,
        ))
    return results
```

### Nested Dependencies

Dependencies can themselves have `depends_on` declarations. The engine resolves the full dependency graph recursively. If pipeline A depends on B and B depends on C, the execution order is: C → B → A.

**Depth limit:** Nested dependency resolution is capped at 3 levels. A pipeline at depth 4 fails with a clear error. This prevents wide-shallow graphs from becoming pathologically expensive (each level creates a full pipeline with its own database connections, plugin instances, and audit records). The limit can be raised if a legitimate use case arises, but for the dogfood, 3 levels is generous.

### Circular Dependency Detection

Before running any dependencies, the engine recursively traverses the full dependency graph by loading `depends_on` from each referenced settings file (and their dependencies, transitively). It builds a directed graph of **canonicalized** settings file paths (using `pathlib.Path.resolve()` to collapse symlinks and relative paths) and checks for cycles using a standard DFS cycle detector. If a cycle is detected, the pipeline fails immediately with a clear error listing the full cycle path. This check is cheap (only reads the `depends_on` key from each config, not the full pipeline) and prevents infinite recursion.

### Idempotency

Dependencies run unconditionally every time the parent pipeline is invoked. There is no "skip if already populated" logic in `depends_on` itself — that concern belongs to commencement gates (future `if_stale` conditions) or external orchestration.

For the dogfood, the recommended pairing is `on_duplicate: overwrite` on the ChromaSink. This makes the indexing pipeline idempotent: re-running it upserts the same documents, producing identical collection state. The audit trail records each re-indexing run separately, which is correct — each run is a distinct operation with its own `run_id` and timing, even if the data is identical.

Operators who want to avoid redundant re-indexing should use external orchestration (e.g., only invoke the query pipeline when the corpus has changed) rather than building conditional execution into `depends_on`. The `depends_on` mechanism is deliberately simple: declare, run, verify.

### Resume Behaviour

On `elspeth resume <run_id>`, dependencies are NOT re-run. Resume picks up the main pipeline from its checkpoint. If the dependencies need re-running (e.g., the collection was corrupted between the original run and the resume), the operator should start a fresh run instead.

### Failure Model

Hard fail only. If a dependency fails, the main pipeline does not start. The operator must fix the dependency issue and rerun. No partial-success thresholds, no automatic retry of dependencies. This is consistent with the "crash on anomaly" philosophy and can be softened later if needed.

### Path Validation

`DependencyConfig.settings` paths are validated at two points:

1. **`elspeth validate` time**: The validator resolves the path and checks that the referenced file exists and is parseable as a valid `ElspethSettings`. This catches typos and missing files before execution. Additionally, if `depends_on` is non-empty and `commencement_gates` is empty, `elspeth validate` emits a warning: "Pipeline declares dependencies but no commencement gates — consider adding a gate to verify dependency output." This is a warning, not an error, to avoid forcing gates on simple pipelines, but it structurally counters the tendency to rely on the readiness contract alone.
2. **Runtime**: The orchestrator re-resolves and loads the dependency settings. This is the authoritative check — the validate-time check is a convenience.

### Signal Handling

If the process receives `SIGINT` or `SIGTERM` during a dependency run, the signal is propagated as `KeyboardInterrupt` to the child orchestrator's `run()` call. The child run terminates (recording its interrupted state in the Landscape), and the parent re-raises the `KeyboardInterrupt` — not a `DependencyFailedError`. This ensures the operator sees "interrupted" rather than "dependency failed," which require different responses.

### Telemetry

The dependency resolution phase emits telemetry spans for each dependency run, including the dependency name, settings path, and outcome (success/failure with `run_id` and duration). This is operational visibility (ephemeral), not audit (the Landscape correlation is the audit record).

## Component 3: Commencement Gates

### Purpose

Evaluate go/no-go conditions after dependencies complete but before the main pipeline starts. Uses expressions evaluated against a pre-flight context.

### Config Shape

New top-level key in pipeline YAML:

```yaml
commencement_gates:
  - name: corpus_ready
    condition: "collections['science-facts']['count'] > 0"
    on_fail: abort
```

- `name` (str, required): Human-readable label. Must be unique within the list.
- `condition` (str, required): Python expression evaluated against the pre-flight context. Must return a truthy/falsy value.
- `on_fail` (Literal["abort"], default "abort"): What to do when the condition is falsy. Only `abort` is supported initially. The field exists so the config shape is forward-compatible with `warn` or other strategies.

### Config Model: `CommencementGateConfig`

Pydantic model added to `core/config.py`:

```python
class CommencementGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    condition: str
    on_fail: Literal["abort"] = "abort"
```

`ElspethSettings` gains an optional field:

```python
commencement_gates: list[CommencementGateConfig] = Field(default_factory=list)
```

### Pre-Flight Context

The engine assembles a context dict that gate expressions evaluate against:

```python
{
    "dependency_runs": {
        "index_corpus": {
            "status": "completed",
            "run_id": "abc-123",
            "duration_ms": 4520,
        }
    },

    "collections": {
        "science-facts": {
            "reachable": True,
            "count": 450,
        }
    },

    "env": {
        "ENVIRONMENT": "production",
        ...
    },
}
```

**`dependency_runs`**: Results from the dependency resolution phase. Keyed by dependency `name`. Empty dict if no `depends_on` declared.

**`collections`**: ChromaDB collection probes, assembled via the `CollectionProbe` protocol (see Collection Probing below).

**`env`**: `os.environ` snapshot. Available for gate expression evaluation but **excluded from the Landscape context_snapshot** to avoid recording secrets.

### Collection Probing — Layer-Safe Injection

The orchestrator (L2) cannot directly construct ChromaDB clients (L3). Collection probing uses a protocol defined in L0, implemented in L3, and injected into L2:

**Protocol** (L0, `contracts/probes.py`):

```python
@dataclass(frozen=True, slots=True)
class CollectionReadinessResult:
    """Unified result type for collection readiness checks.

    Used by both CollectionProbe (commencement gates) and
    RetrievalProvider.check_readiness() (transform pre-condition).
    """
    collection: str
    reachable: bool
    count: int
    message: str  # Human-readable: "Collection 'X' has 450 documents"

@runtime_checkable
class CollectionProbe(Protocol):
    """Probes a vector store collection for readiness."""
    collection_name: str

    def probe(self) -> CollectionReadinessResult: ...
```

All fields on `CollectionReadinessResult` are scalars — no `__post_init__` freeze guard needed. This type is defined in L0 (`contracts/probes.py`) and used by both the `CollectionProbe` protocol and `RetrievalProvider.check_readiness()`, eliminating the previous duplication between `CollectionProbeResult` (L0) and `ReadinessResult` (L3).

**Implementations** (L3, `plugins/infrastructure/clients/retrieval/`):
- `ChromaCollectionProbe`: Constructs a ChromaDB client, calls `collection.count()`.
- `AzureSearchCollectionProbe`: Calls the Azure Search count endpoint.

**Explicit probe declarations** (pipeline YAML):

Collection probes are explicitly declared in the pipeline config, not auto-discovered from plugin configs. This avoids implicit coupling where a factory must understand the internal config shape of every vector-store plugin.

```yaml
collection_probes:
  - collection: science-facts
    provider: chroma
    provider_config:
      mode: persistent
      persist_directory: ./chroma_data
```

`ElspethSettings` gains an optional field:

```python
collection_probes: list[CollectionProbeConfig] = Field(default_factory=list)
```

`CollectionProbeConfig` is a Pydantic model in `core/config.py` with `collection` (str), `provider` (str), and `provider_config` (dict). The connection fields in `provider_config` reuse the shared `ChromaConnectionConfig` (see Config Symmetry below).

**Assembly** (L3, `plugins/infrastructure/`):
A factory function `build_collection_probes(probe_configs: list[CollectionProbeConfig]) -> list[CollectionProbe]` constructs a probe for each declared config. This function lives in L3 and is called by the orchestrator before gate evaluation. It does not scan plugin configs — the operator explicitly declares what to probe.

**Injection into L2:**
The orchestrator calls `build_collection_probes()` and receives a list of `CollectionProbe` protocol objects. It calls `probe.probe()` on each and assembles the `collections` context dict. The orchestrator never imports ChromaDB or any L3 client — it only knows the L0 protocol.

If a probe raises an exception (collection unreachable), the result is `CollectionProbeResult(collection=name, reachable=False, count=0)`. Probe failures do not abort the pipeline directly. However, a gate condition like `collections['science-facts']['count'] > 0` will evaluate to `False` when `count=0` (whether the collection is empty or unreachable), causing the gate to fail. The effective behaviour is that unreachable collections fail count-based gates — this is by design, not an accident.

**Canonical gate expression idiom:** Use `collections['name']['count'] > 0` as the standard form. The `reachable` field is informational (useful for diagnostics and error messages) but gate expressions should not need to consult it separately — `count > 0` inherently requires reachability because unreachable collections produce `count=0`. Writing `count > 0 and reachable` is redundant.

### Expression Evaluation

Gate expressions are evaluated using ELSPETH's existing `ExpressionParser` (`core/expression_parser.py`), not `eval()`. The parser uses AST node whitelisting — it parses the expression with `ast.parse()`, walks the AST, and rejects any node type not in an explicit allow-list. This eliminates `eval()` bypass vectors (e.g., `.__class__.__bases__` traversal) at parse time.

**Allowed AST nodes:** `Expression`, `BoolOp`, `Compare`, `Subscript`, `Name`, `Constant`, `And`, `Or`, `Not`, `Gt`, `Lt`, `GtE`, `LtE`, `Eq`, `NotEq`. No `Attribute`, no `Call`, no `Import`.

**Extension required:** The existing parser evaluates against `PipelineRow` objects. It needs a minor extension to accept a plain dict context for commencement gate evaluation. The allowed node set and evaluation mechanics are the same — only the namespace binding changes.

**Validation at config load time:** Gate expressions are parsed and AST-validated during `CommencementGateConfig` construction (Pydantic `@model_validator`). If the AST contains disallowed nodes, a `ValueError` is raised — Pydantic converts that to a validation error. This means `elspeth validate` catches malformed expressions before execution, not mid-run.

If the expression raises an exception during evaluation, the gate fails with the exception message included in the error. This is intentional — a gate expression that can't evaluate is a configuration error, not a pass.

### Execution

Gates are evaluated sequentially in declared order, after dependency resolution, before database initialization.

```python
def _evaluate_commencement_gates(
    self,
    gates: list[CommencementGateConfig],
    context: dict[str, Any],
) -> list[GateResult]:
    # Freeze the entire context before evaluation to close the TOCTOU window.
    # Between context assembly and gate evaluation, no mutation is possible.
    frozen_context = deep_freeze(context)
    audit_snapshot = _build_audit_snapshot(frozen_context)

    results = []
    for gate in gates:
        try:
            parser = ExpressionParser(gate.condition)
            passed = bool(parser.evaluate(frozen_context))
        except Exception as exc:
            raise CommencementGateFailedError(
                gate_name=gate.name,
                condition=gate.condition,
                reason=f"Expression raised {type(exc).__name__}: {exc}",
                context_snapshot=audit_snapshot,
            ) from exc

        if not passed:
            raise CommencementGateFailedError(
                gate_name=gate.name,
                condition=gate.condition,
                reason="Condition evaluated to falsy",
                context_snapshot=audit_snapshot,
            )

        results.append(GateResult(
            name=gate.name,
            condition=gate.condition,
            result=True,
            context_snapshot=audit_snapshot,
        ))
    return results
```

### Context Snapshot for Audit

The context snapshot recorded in the Landscape includes `dependency_runs` and `collections` but **excludes `env`** (may contain secrets). The snapshot is deep-frozen via `deep_freeze()` at the point of capture, before being passed to `CommencementGateFailedError` or stored as run metadata. This ensures the dict cannot be mutated between capture and Landscape write — it is Tier 1 data from the moment of capture.

```python
def _build_audit_snapshot(context: dict[str, Any]) -> Mapping[str, Any]:
    """Build a frozen context snapshot for audit, excluding env."""
    snapshot = {
        "dependency_runs": context["dependency_runs"],
        "collections": context["collections"],
    }
    return deep_freeze(snapshot)
```

### `GateResult`

Frozen dataclass in `core/config.py`:

```python
@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    condition: str
    result: bool
    context_snapshot: Mapping[str, Any]

    def __post_init__(self) -> None:
        freeze_fields(self, "context_snapshot")
```

### `CommencementGateFailedError`

Plain exception subclass in `contracts/errors.py` (not a dataclass). Receives `context_snapshot` as a deep-frozen `Mapping` — the caller is responsible for freezing before construction (enforced by `_build_audit_snapshot()`).

### Landscape Record

Gate results are included in the main pipeline's run metadata:

```python
{
    "commencement_gates": [
        {
            "name": "corpus_ready",
            "condition": "collections['science-facts']['count'] > 0",
            "result": true,
            "context_snapshot": {
                "dependency_runs": {"index_corpus": {"status": "completed", ...}},
                "collections": {"science-facts": {"count": 450, "reachable": true}}
            }
        }
    ]
}
```

The snapshot provides full auditability of the go/no-go decision. Environment variables are excluded to prevent secret leakage into the audit trail.

### Telemetry

The commencement gate evaluation phase emits telemetry spans for each gate, including the gate name, condition expression, and result. This is operational visibility alongside the dependency resolution telemetry spans.

## Component 4: RAG Retrieval Readiness Contract

### Purpose

The existing `RAGRetrievalTransform` gains a readiness check in `on_start()` that verifies the target collection exists and has documents. If the check fails, the pipeline crashes immediately.

### Provider Protocol Extension

`RetrievalProvider` gains a new method returning the unified `CollectionReadinessResult` from L0:

```python
@runtime_checkable
class RetrievalProvider(Protocol):
    def search(self, query: str, top_k: int, min_score: float,
               *, state_id: str, token_id: str) -> list[RetrievalChunk]: ...

    def check_readiness(self) -> CollectionReadinessResult: ...
```

`CollectionReadinessResult` (defined in `contracts/probes.py`, see Component 3) is the single result type for all collection readiness checks — used by both `CollectionProbe.probe()` and `RetrievalProvider.check_readiness()`. Fields:

- `collection`: Collection name that was checked.
- `reachable`: `True` if the collection endpoint responded.
- `count`: Number of documents found (0 if collection doesn't exist or is unreachable).
- `message`: Human-readable status. Examples:
  - `"Collection 'science-facts' has 450 documents"`
  - `"Collection 'science-facts' not found"`
  - `"Collection 'science-facts' is empty"`

### Provider Implementations

**ChromaSearchProvider.check_readiness():**
- Get collection by name. If not found, return `ReadinessResult(ready=False, ..., message="Collection '...' not found")`.
- Call `collection.count()`. If 0, return not ready. Otherwise return ready with count.

**AzureSearchProvider.check_readiness():**
- `GET /indexes/{index_name}/docs/$count` via the audited HTTP client.
- If index not found (404), return not ready.
- If count is 0, return not ready.
- Otherwise return ready with count.

Both implementations wrap external calls with the same error handling as `search()` — transient errors are retryable, permanent errors are not.

`check_readiness()` is a single-attempt call — it does not go through the engine's retry machinery. It runs during `on_start()`, which is a setup phase, not a row-processing phase. If a transient network error causes the readiness check to fail, the pipeline fails to start, and the operator retries the entire run. This is acceptable for a pre-condition check — retrying silently would mask infrastructure problems that the operator should know about.

All existing `RetrievalProvider` implementations (`ChromaSearchProvider`, `AzureSearchProvider`) must be updated with `check_readiness()` in the same commit as the protocol extension.

### Transform Integration

In `RAGRetrievalTransform.on_start()`, after constructing the provider:

```python
def on_start(self, ctx: LifecycleContext) -> None:
    super().on_start(ctx)
    # ... existing provider construction ...

    result = self._provider.check_readiness()
    if result.count == 0:
        raise RetrievalNotReadyError(
            f"RAG transform '{self.name}' requires a populated collection. "
            f"{result.message}"
        )
```

`RetrievalNotReadyError` inherits from the base ELSPETH error hierarchy. It is:
- Not retryable — this is a configuration/sequencing error, not a transient failure.
- Not a row-level error — it prevents the pipeline from starting.
- Includes the collection name and document count in the error message.

### Defence in Depth

Commencement gates and the readiness contract serve different purposes:

| Layer | Catches | Fires when | Configured by |
|-------|---------|-----------|--------------|
| Commencement gate | "Collection has data" | Before `on_start()`, pre-flight | Pipeline operator |
| Readiness contract | "My provider can reach the collection and it has data" | During `on_start()` | Always (built into transform) |

A commencement gate catches the problem earlier and with a better error message (includes the expression and context snapshot). The readiness contract catches it even when no commencement gate is configured.

An operator using `depends_on` + commencement gates gets the full experience. An operator who pre-populates the collection externally and skips `depends_on` still gets the readiness check.

## End-to-End Example

### File Layout

```
examples/chroma_rag_indexed/
  index_pipeline.yaml
  query_pipeline.yaml
  documents.csv
  questions.csv
  chroma_data/
  runs/
    audit.db
  output/
    results.jsonl
    quarantined.jsonl
```

### Indexing Pipeline (`index_pipeline.yaml`)

```yaml
# No transforms — documents are written directly from source to ChromaSink.
# A chunking transform would be added here for long-document corpora.
source:
  plugin: csv_source
  on_success: output
  options:
    file_path: ./documents.csv
    schema:
      mode: fixed
      fields:
        - "doc_id: str"
        - "text_content: str"
        - "topic: str"
        - "subtopic: str"

sinks:
  output:
    plugin: chroma_sink
    options:
      collection: science-facts
      mode: persistent
      persist_directory: ./chroma_data
      distance_function: cosine
      field_mapping:
        document: text_content
        id: doc_id
        metadata:
          - topic
          - subtopic
      on_duplicate: overwrite
      schema:
        mode: fixed
        fields:
          - "doc_id: str"
          - "text_content: str"
          - "topic: str"
          - "subtopic: str"

landscape:
  url: sqlite:///./runs/audit.db
```

### Query Pipeline (`query_pipeline.yaml`)

```yaml
depends_on:
  - name: index_corpus
    settings: ./index_pipeline.yaml

collection_probes:
  - collection: science-facts
    provider: chroma
    provider_config:
      mode: persistent
      persist_directory: ./chroma_data

commencement_gates:
  - name: corpus_ready
    condition: "collections['science-facts']['count'] > 0"
    on_fail: abort

source:
  plugin: csv_source
  on_success: retrieve
  options:
    file_path: ./questions.csv
    schema:
      mode: fixed
      fields:
        - "question: str"

transforms:
  - name: retrieve
    plugin: rag_retrieval
    input: source_output
    on_success: output
    on_error: quarantine
    options:
      query_field: question
      output_prefix: sci
      provider: chroma
      provider_config:
        collection: science-facts
        mode: persistent
        persist_directory: ./chroma_data
        distance_function: cosine
      top_k: 3
      min_score: 0.0
      on_no_results: quarantine
      context_format: numbered
      schema:
        mode: flexible

sinks:
  output:
    plugin: json_sink
    options:
      file_path: ./output/results.jsonl
      format: jsonl
      schema:
        mode: flexible

  quarantine:
    plugin: json_sink
    options:
      file_path: ./output/quarantined.jsonl
      format: jsonl
      schema:
        mode: flexible

landscape:
  url: sqlite:///./runs/audit.db
```

### Execution Flow

```
$ elspeth run --settings query_pipeline.yaml --execute

1. Dependency resolution
   → Load index_pipeline.yaml
   → Create child Orchestrator, run indexing pipeline
   → CSV source emits document rows
   → ChromaSink upserts documents into 'science-facts'
   → Indexing run abc-123 completes successfully

2. Pre-flight context assembly
   → Probe collection 'science-facts': 10 documents found
   → Build context: {dependency_runs: {...}, collections: {...}, env: {...}}

3. Commencement gate evaluation
   → 'corpus_ready': collections['science-facts']['count'] > 0 → True ✓

4. Main pipeline starts (run def-456)
   → Run metadata: {dependency_runs: [...], commencement_gates: [...]}

5. RAG transform on_start()
   → Provider constructed
   → Readiness check: 'science-facts' has 10 documents ✓

6. Row processing
   → Questions retrieved, augmented with context, written to output

7. Run def-456 completes
```

### Audit Trail

For any output row, an auditor can trace:

1. **Row lineage**: `explain(recorder, "def-456", token_id)` — full DAG path
2. **Retrieved context**: RAG transform call records — which chunks, what scores
3. **Corpus state**: Commencement gate snapshot — `count=10` at pipeline start
4. **Corpus provenance**: `dependency_runs` links to indexing run `abc-123`
5. **Indexed documents**: Every document in `abc-123` has call records with canonical content hashes
6. **Go/no-go decision**: Gate expression, evaluation result, and context snapshot

## Testing Strategy

**Critical requirement:** Integration tests MUST use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()` per CLAUDE.md. Tests that construct plugins or orchestrators from hand-built objects bypass production code paths and are not valid integration tests.

**ChromaDB test mode:** All integration tests use ChromaDB persistent mode with `tmp_path` fixture (pytest). This is deterministic, requires no running server, and works in all CI environments. Client mode against a container is not required for the dogfood.

### Unit Tests

**ChromaSink config:**
- Field mapping required — missing `field_mapping` raises validation error.
- Mode/connection mutual exclusion — `persist_directory` with `mode=client` raises.
- `field_mapping.document` pointing to a non-`str` schema field raises.
- `field_mapping.metadata` field with `datetime` type raises (must be str/int/float/bool).
- `field_mapping.metadata` field not in schema raises.
- `on_duplicate` values — only `overwrite`, `skip`, `error` accepted.

**ChromaSink write:**
- `write()` with mocked ChromaDB client — verify correct API calls, content hashing, artifact descriptor.
- `on_duplicate: overwrite` — calls `collection.upsert()`.
- `on_duplicate: skip` — calls `collection.get()` to find existing IDs, then `collection.add()` only for new ones. Verify skipped IDs appear in `record_call()` request_data.
- `on_duplicate: error` with all-new IDs — calls `collection.get()`, finds none, proceeds with `add()`.
- `on_duplicate: error` with partial overlap (3 of 5 IDs exist) — raises pipeline-level error listing only the duplicate IDs. No write occurs.
- Audit recording — verify `record_call()` invoked with correct `provider`, `call_type`, `row_count`, `document_ids`.
- `AuditIntegrityError` path — mock `record_call()` to raise after successful `collection.upsert()`. Verify `AuditIntegrityError` propagates (not swallowed).
- `flush()` — verify no ChromaDB client method is called (no-op assertion via mock call count).

**Dependency resolution:**
- Single dependency, success — `bootstrap_and_run()` called with resolved path, `DependencyRunResult` constructed with `indexed_at`.
- Single dependency, failure — `DependencyFailedError` raised with dependency name and run_id.
- Multiple dependencies — sequential execution verified by call order.
- Circular detection (self-loop) — A→A detected.
- Circular detection (2-hop) — A→B→A detected.
- Circular detection (3-hop) — A→B→C→A detected.
- Depth limit exceeded — 4-level nesting raises clear error.
- Path resolution — relative paths resolved from parent config directory using `pathlib.Path.resolve()`.
- `KeyboardInterrupt` during dependency — propagated as interrupt, not `DependencyFailedError`.
- Dependency resolution seam: `bootstrap_and_run()` is injectable (the function, not the orchestrator class) so unit tests substitute a stub that returns a `RunResult` without running a real pipeline.

**Commencement gates:**
- Gate passes — `GateResult` recorded with frozen context snapshot.
- Gate fails — `CommencementGateFailedError` with context snapshot. Verify `reachable=False` included in error reason when `count=0` due to unreachable collection.
- Expression error — verify error includes exception details and gate name.
- AST validation — disallowed nodes (`__import__('os')`, `open('/etc/passwd')`, `print('x')`, `().__class__.__bases__`) rejected at config validation time, not at evaluation time. Assert `ValueError` from Pydantic, not `CommencementGateFailedError`.
- Context snapshot excludes `env` — build context with `env: {"SECRET_KEY": "abc123"}`, run `_build_audit_snapshot()`, assert `SECRET_KEY` not in result.
- Context is frozen before evaluation — verify `deep_freeze()` called on context before `ExpressionParser.evaluate()`.
- Hypothesis property test: generate arbitrary strings as gate expressions. Verify they either evaluate to truthy/falsy or raise — never produce side effects observable outside the namespace.

**Readiness contract:**
- Collection exists with documents — ready (count > 0, reachable = True).
- Collection exists but empty — not ready, message includes "is empty".
- Collection doesn't exist — not ready, message includes "not found".
- Provider unreachable — error propagation (single-attempt, no retry).
- All existing `RetrievalProvider` test doubles updated with `check_readiness()` — use `spec_set` not `spec` on mocks to catch missing methods.

**Collection probes:**
- `build_collection_probes()` constructs probes from explicit config declarations.
- Probe success — `CollectionReadinessResult` with correct count.
- Probe failure (unreachable) — `CollectionReadinessResult(reachable=False, count=0)`.

### Integration Tests

**ChromaSink lifecycle** (focused):
- Full pipeline: CSV source → ChromaSink with real ephemeral ChromaDB (persistent mode, `tmp_path`).
- Uses `instantiate_plugins_from_config()` and `ExecutionGraph.from_plugin_instances()`.
- Verify documents written to collection with correct IDs, content, metadata.
- **Audit trail assertion:** Query Landscape for `record_call` entries — verify `provider="chromadb"`, correct `row_count`, content hash matches canonical JSON of input.

**depends_on** (focused):
- Two-pipeline integration: indexing pipeline populates collection, query pipeline retrieves from it.
- Uses `bootstrap_and_run()` for the dependency (same code path as CLI).
- **Audit trail assertion:** Query run metadata — verify `dependency_runs` contains child `run_id` and `indexed_at` timestamp.
- Verify indexing run's Landscape has `record_call` entries for ChromaSink writes.

**Commencement gates** (focused):
- Gate evaluates against pre-populated collection. Gate passes.
- **Audit trail assertion:** Query run metadata — verify `commencement_gates` entry with correct condition, result, and context_snapshot. Verify snapshot does NOT contain `env`.

**Readiness contract** (focused):
- RAG transform against empty collection — `RetrievalNotReadyError` raised, pipeline never processes rows.
- RAG transform against populated collection — readiness check passes, rows processed normally.

**End-to-end smoke test:**
- Full `depends_on` + commencement gate + RAG retrieval pipeline.
- Verify all three mechanisms fire in order: dependency runs, gate evaluates, readiness checks, rows processed.

**Failure paths:**
- Dependency fails → main pipeline never starts. Landscape records the failed dependency run.
- Gate fails → main pipeline never starts. Error includes gate name, condition, context snapshot.
- Empty collection, no `depends_on`, no gate → readiness contract catches it.
- Resume after interruption — dependencies are NOT re-run. Verify no second child `run_id` in Landscape.

**Nested dependency ordering:**
- A depends on B, B depends on C. Verify execution order C→B→A via Landscape `run_id` timestamps.

## Future Extensions

These are explicitly out of scope for this design but inform the shape of what we're building:

### P1 Follow-On (implement soon after this design ships)

- **`if_stale` dependency condition**: Run the dependency only if the collection hasn't been updated since a threshold. The `indexed_at` timestamp on `DependencyRunResult` provides the foundation — this extension adds a `condition` field to `DependencyConfig` that evaluates against the last `indexed_at` for the same `settings_hash`. Without this, the "Fixes that Fail" dynamic identified in systems review will push operators to bypass `depends_on` as corpus size grows, re-opening the audit gap.

### P2 (build when needed)

- **Chunking transform**: A transform that splits long documents into overlapping segments. Would sit between the source and ChromaSink in the indexing pipeline.
- **Additional commencement gate conditions**: `warn` (log but proceed), service health probes, Landscape queries ("last successful indexing run was < 24h ago").
- **Multi-source DAGs (Approach C)**: Eventually, the dependency pipeline could be compiled into the main pipeline's DAG as prefix stages. This requires breaking the one-source-per-run invariant.
- **Azure Search sink**: Same pattern as ChromaSink but targeting Azure AI Search indexes.
