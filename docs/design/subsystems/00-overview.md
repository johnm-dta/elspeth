# ELSPETH Subsystems Overview

**Version:** 1.0
**Date:** 2026-01-12
**Status:** Design

---

## Purpose

This document provides a high-level map of all subsystems in ELSPETH, their responsibilities, and how they interact. Each subsystem will have its own detailed design document.

---

## Subsystem Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                    CLI                                       │
│                           (User interaction layer)                           │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONFIGURATION                                   │
│                    (Load, validate, resolve settings)                        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               SDA ENGINE                                     │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │   SENSE     │───▶│   DECIDE    │───▶│    ACT      │                      │
│  │  (Sources)  │    │ (Transforms)│    │   (Sinks)   │                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
│         │                  │                  │                              │
│         └──────────────────┼──────────────────┘                              │
│                            ▼                                                 │
│                    ┌─────────────┐                                           │
│                    │ Orchestrator│                                           │
│                    └─────────────┘                                           │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│    PLUGIN SYSTEM    │  │      LANDSCAPE      │  │    PAYLOAD STORE    │
│                     │  │                     │  │                     │
│ - Hookspecs         │  │ - Run tracking      │  │ - Large blob store  │
│ - Registration      │  │ - Row states        │  │ - Retention         │
│ - Base classes      │  │ - External calls    │  │ - Compression       │
│ - Discovery         │  │ - Routing decisions │  │                     │
└─────────────────────┘  │ - Artifacts         │  └─────────────────────┘
                         │ - Queries/Explain   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     CANONICAL       │
                         │                     │
                         │ - JSON normalization│
                         │ - Hashing           │
                         │ - Versioning        │
                         └─────────────────────┘
```

---

## Subsystem Summaries

### 1. Landscape

**Intent:** The audit backbone. Records everything that happens so any output can be traced to its source.

**Responsibilities:**

- Track runs with resolved configuration and reproducibility grade
- Register plugin instances (nodes in the execution graph)
- **Track tokens** (row instances flowing through the DAG)
- Record node states (what happened at each node for each token)
- Capture external calls (LLM, HTTP, ML) with request/response
- Record routing events with edge selection and reason
- Track aggregation batches with membership
- Register artifacts produced by sinks
- Answer `explain()` queries with complete lineage

**Core Tables (DAG-Aware):**

```sql
-- Runs and configuration
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    config_hash TEXT NOT NULL,
    settings_json TEXT NOT NULL,           -- Resolved config stored, not just hash
    reproducibility_grade TEXT,            -- FULL_REPRODUCIBLE, REPLAY_REPRODUCIBLE, ATTRIBUTABLE_ONLY
    canonical_version TEXT NOT NULL,
    status TEXT NOT NULL
);

-- Nodes in the execution graph (plugin instances)
CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    plugin_name TEXT NOT NULL,
    node_type TEXT NOT NULL,               -- source, transform, gate, aggregation, coalesce, sink
    plugin_version TEXT NOT NULL,
    determinism TEXT NOT NULL,  -- deterministic, seeded, io_read, io_write, external_call, non_deterministic
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    schema_hash TEXT,                      -- Input/output schema fingerprint
    sequence_in_pipeline INTEGER,
    registered_at TIMESTAMP NOT NULL
);

-- Edges: static graph structure (validated at config time)
CREATE TABLE edges (
    edge_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    from_node_id TEXT NOT NULL REFERENCES nodes(node_id),
    to_node_id TEXT NOT NULL REFERENCES nodes(node_id),
    label TEXT NOT NULL,                   -- "continue", "suspicious", "stats_branch"
    default_mode TEXT NOT NULL,            -- move, copy
    created_at TIMESTAMP NOT NULL,
    UNIQUE(run_id, from_node_id, label)
);

-- Source rows (original identity)
CREATE TABLE rows (
    row_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    source_node_id TEXT NOT NULL REFERENCES nodes(node_id),
    row_index INTEGER NOT NULL,
    source_data_hash TEXT NOT NULL,
    source_data_ref TEXT,                  -- Payload store reference
    created_at TIMESTAMP NOT NULL,
    UNIQUE(run_id, row_index)
);

-- Tokens: row instances flowing through DAG paths
CREATE TABLE tokens (
    token_id TEXT PRIMARY KEY,
    row_id TEXT NOT NULL REFERENCES rows(row_id),
    fork_group_id TEXT,                    -- Links tokens from same fork decision
    join_group_id TEXT,                    -- Links tokens merged in same coalesce
    branch_name TEXT,                      -- Which branch this token took
    created_at TIMESTAMP NOT NULL
);

-- Token parents: supports multi-parent joins (coalesce)
CREATE TABLE token_parents (
    token_id TEXT NOT NULL REFERENCES tokens(token_id),
    parent_token_id TEXT NOT NULL REFERENCES tokens(token_id),
    ordinal INTEGER NOT NULL,              -- Order of parents (deterministic)
    PRIMARY KEY (token_id, parent_token_id),
    UNIQUE (token_id, ordinal)
);

-- Node states: what happened when a token visited a node
CREATE TABLE node_states (
    state_id TEXT PRIMARY KEY,
    token_id TEXT NOT NULL REFERENCES tokens(token_id),
    node_id TEXT NOT NULL REFERENCES nodes(node_id),
    step_index INTEGER NOT NULL,           -- Position in token's execution path
    attempt INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,                  -- open, completed, failed
    input_hash TEXT NOT NULL,
    output_hash TEXT,                      -- May be NULL on failure
    context_before_json TEXT,
    context_after_json TEXT,
    duration_ms REAL,
    error_json TEXT,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    UNIQUE(token_id, node_id, attempt),
    UNIQUE(token_id, step_index)           -- No duplicate positions in path
);

-- Routing events: edge selection decisions
CREATE TABLE routing_events (
    event_id TEXT PRIMARY KEY,
    state_id TEXT NOT NULL REFERENCES node_states(state_id),
    edge_id TEXT NOT NULL REFERENCES edges(edge_id),  -- Which edge was taken
    routing_group_id TEXT NOT NULL,        -- Links actions from same decision
    ordinal INTEGER NOT NULL,              -- Order within group
    mode TEXT NOT NULL,                    -- move, copy (may override edge default)
    reason_hash TEXT,
    reason_ref TEXT,                       -- Payload store reference
    created_at TIMESTAMP NOT NULL,
    UNIQUE(routing_group_id, ordinal)
);

-- External calls within a node state
CREATE TABLE calls (
    call_id TEXT PRIMARY KEY,
    state_id TEXT NOT NULL REFERENCES node_states(state_id),
    call_index INTEGER NOT NULL,
    call_type TEXT NOT NULL,               -- llm, http, sql, filesystem
    status TEXT NOT NULL,                  -- success, error
    request_hash TEXT NOT NULL,
    request_ref TEXT,
    response_hash TEXT,                    -- NULL if status=error (no response received)
    response_ref TEXT,
    error_json TEXT,                       -- Error details if status=error
    latency_ms REAL,                       -- May be NULL if timeout
    created_at TIMESTAMP NOT NULL,
    UNIQUE(state_id, call_index)
);

-- Aggregation batches (draft → executing → completed/failed)
CREATE TABLE batches (
    batch_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    aggregation_node_id TEXT NOT NULL REFERENCES nodes(node_id),
    aggregation_state_id TEXT REFERENCES node_states(state_id),  -- Links to processing span (NULL while draft)
    trigger_reason TEXT,                   -- NULL while draft, set when flush triggered
    attempt INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('draft', 'executing', 'completed', 'failed')),
    created_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);

CREATE TABLE batch_members (
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    token_id TEXT NOT NULL REFERENCES tokens(token_id),
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (batch_id, token_id),
    UNIQUE(batch_id, ordinal)              -- Enforce ordinal uniqueness
);

CREATE TABLE batch_outputs (
    batch_output_id TEXT PRIMARY KEY,      -- Surrogate key for simpler joins
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    output_type TEXT NOT NULL,             -- token, artifact
    output_id TEXT NOT NULL,               -- token_id or artifact_id
    UNIQUE(batch_id, output_type, output_id)  -- Prevent duplicate outputs
);

-- Artifacts produced by sinks
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    produced_by_state_id TEXT NOT NULL REFERENCES node_states(state_id),
    sink_node_id TEXT NOT NULL REFERENCES nodes(node_id),
    artifact_type TEXT NOT NULL,
    path_or_uri TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL
);
```

**Status Vocabulary (Two Distinct Concepts):**

There are two status systems that must not be confused:

1. **`node_states.status`** - Processing status at a single node:

   | Status | Meaning |
   |--------|---------|
   | `open` | Transform is currently executing |
   | `completed` | Transform finished successfully |
   | `failed` | Transform failed (may be retried) |

2. **Token Terminal States** - Final disposition of a token in the pipeline (derived, not stored):

   | State | Meaning | How Derived |
   |-------|---------|-------------|
   | `COMPLETED` | Reached output sink | Last node_state is at a sink node |
   | `ROUTED` | Sent to named sink by gate | routing_event with move mode to sink |
   | `FORKED` | Split into child tokens | Has child tokens in token_parents |
   | `CONSUMED_IN_BATCH` | Fed into aggregation | Exists in batch_members |
   | `COALESCED` | Merged with other tokens | Is parent in token_parents for join |
   | `QUARANTINED` | Failed, stored for investigation | Last node_state.status = failed + quarantine flag |
   | `FAILED` | Failed, not recoverable | Last node_state.status = failed, no quarantine |

**Key insight:** Token terminal states are *derived* from the combination of node_states, routing_events, and batch membership—not stored as a column. This avoids redundant state that could become inconsistent.

**Key Principle:** No silent drops. Every token has a terminal state. Every decision is recorded.

**Depends on:** Canonical (for hashing), Payload Store (for large blobs)

---

### 2. Plugin System

**Intent:** Extensibility without modifying core. All data processing happens through plugins.

**Responsibilities:**

- Define hookspecs for Source, Transform, Sink
- Plugin discovery and registration
- Plugin instance lifecycle management
- Configuration validation per plugin type
- Base classes for common patterns

**Plugin Primitives:**

| Primitive | Role | Cardinality |
|-----------|------|-------------|
| **Source** | Get data into the system | One per run |
| **Transform** | Process/classify/route data | Zero or more, ordered |
| **Sink** | Output data somewhere | One or more, named |

**Transform Categories:**

| Category | Behavior | State | Example |
|----------|----------|-------|---------|
| **Row Transform** | Process one row, emit one row | Stateless | Field mapping, validation |
| **Row Gate** | Process one row, decide destination(s) | Stateless | Threshold check, classifier |
| **Aggregation** | Accumulate N rows, emit 1 result | Stateful | Statistics, batch summary |
| **Coalesce** | Merge results from parallel paths | Stateful | Combine parallel aggregation results |

**Coalesce Policies:**

Coalesce nodes must declare how to handle partial arrivals:

| Policy | Behavior |
|--------|----------|
| `require_all` | Barrier - wait for all branches; any failure fails the coalesce |
| `quorum(n)` | Merge if ≥ n branches succeed; missing branches recorded explicitly |
| `best_effort` | Merge whatever arrives by timeout; route missing to quarantine |

**Coalesce Correlation Key:**

How does coalesce know which branch outputs belong together?

| Key Strategy | Use Case |
|--------------|----------|
| `singleton` | Default - merge the only outputs (for per-run aggregation) |
| `parent_token_id` | Classic fork/join - match by original forked token |
| `row_id` | Match by source row identity |
| `field(name)` | Match by specific field value (advanced) |

```yaml
- plugin: coalesce
  policy: require_all
  key: parent_token_id           # How to correlate branch outputs
  inputs:
    - path: stats_branch
    - path: classifier_branch
  merge_strategy:
    mean: take_from(stats_branch)
    category: take_from(classifier_branch)
```

**Coalesce Audit Record:**

| Field | Purpose |
|-------|---------|
| `correlation_key` | What key strategy was used |
| `expected_branches` | Which paths were expected |
| `arrived_branches` | Which paths completed |
| `failed_branches` | Which paths failed or timed out |
| `policy_applied` | Which policy decided the outcome |

**Routing Semantics:**

Gates can route to **one or more** destinations:

| Routing | Behavior | Example |
|---------|----------|---------|
| `route: A` | Send to sink A, exit pipeline | "Flagged for review" |
| `route: [A, B, C]` | Copy to A, B, and C in parallel | "Send to stats, classifier, and archive" |
| `continue` | Proceed to next transform | Normal flow |

Multi-destination routing ("split") is just routing to multiple places - not a separate plugin type.

**Parallel Aggregation Pattern:**

Aggregations collapse N rows → 1 row. You cannot chain aggregations (nothing left to aggregate). But you CAN route to multiple aggregations in parallel:

```
                         ┌─── Aggregation A (mean) ───┐
                         │                            │
Input rows ── Gate ──────┼─── Aggregation B (max)  ───┼── Coalesce ── Output
             route:[A,B,C]                            │
                         └─── Aggregation C (count) ──┘
```

Each aggregation receives the same input rows, produces one result, and coalesce merges them.

**Aggregation Semantics:**

Aggregation plugins accumulate rows until one of:

1. **Threshold reached** - e.g., 100 rows collected
2. **Flush signal** - source indicates no more rows coming
3. **Timeout** - configured max wait time
4. **Trigger condition** - custom logic (e.g., group key changes)

**Batch State Machine (Draft → Executing → Terminal):**

```
                    accept()                    flush triggered
    ┌─────────┐    first row     ┌─────────┐                    ┌────────────┐
    │ (none)  │ ────────────────▶│  DRAFT  │ ──────────────────▶│ EXECUTING  │
    └─────────┘                  └─────────┘                    └────────────┘
                                      │                               │
                                      │ accept()                      │
                                      │ subsequent rows               │
                                      ▼                               │
                                 (add to batch_members)               │
                                                                      │
                                              ┌───────────────────────┴───────────────────────┐
                                              │                                               │
                                              ▼                                               ▼
                                       ┌────────────┐                                  ┌────────────┐
                                       │ COMPLETED  │                                  │  FAILED    │
                                       └────────────┘                                  └────────────┘
```

**Design principle:** Persist membership on every `accept()`, not just at flush. This means:

- No custom checkpointing logic needed
- Crash recovery is a query, not deserialization
- "What's pending in this aggregation?" is always answerable

```python
class AggregationTransform:
    """Accumulates rows until trigger, then processes batch."""

    def __init__(self):
        self._batch_id: str | None = None
        self._buffer: list[dict] = []  # In-memory for processing

    def accept(self, row: dict, token_id: str, context: dict) -> AcceptResult:
        """Accept a row into the accumulator.

        Creates draft batch on first row, persists membership immediately.
        """
        # Create draft batch on first row
        if self._batch_id is None:
            self._batch_id = landscape.create_batch(
                node_id=self.node_id,
                status="draft",
            )

        # Persist membership immediately (crash-safe)
        landscape.add_batch_member(
            batch_id=self._batch_id,
            token_id=token_id,
            ordinal=len(self._buffer),
        )

        self._buffer.append(row)

        if self._should_trigger():
            return AcceptResult(accepted=True, trigger=True)
        return AcceptResult(accepted=True, trigger=False)

    def flush(self, context: dict) -> list[dict]:
        """Process accumulated rows and return results."""
        # Transition to executing
        landscape.update_batch_status(self._batch_id, "executing")

        try:
            result = self._compute_aggregate(self._buffer)
            landscape.update_batch_status(self._batch_id, "completed")
            self._reset()
            return [result]
        except Exception as e:
            landscape.update_batch_status(self._batch_id, "failed", error=str(e))
            raise

    def _reset(self):
        self._batch_id = None
        self._buffer = []
```

**Crash Recovery:**

```python
def restore_aggregations_on_startup(run_id: str, aggregation_plugins: dict):
    """Restore draft batches to their aggregation plugins after crash."""
    # Restore draft batches (accepting rows, not yet triggered)
    for batch in landscape.get_batches(run_id, status="draft"):
        plugin = aggregation_plugins[batch.node_id]
        members = landscape.get_batch_members(batch.batch_id)

        plugin._batch_id = batch.batch_id
        plugin._buffer = [
            payload_store.get(m.row_data_ref)
            for m in sorted(members, key=lambda m: m.ordinal)
        ]

    # Handle executing batches (crashed mid-flush) - mark failed for retry
    for batch in landscape.get_batches(run_id, status="executing"):
        landscape.update_batch_status(
            batch.batch_id, "failed",
            error="Process crashed during flush execution"
        )
```

**Key invariant:** Rows are never "lost in limbo" - they're attached to a draft batch from the moment they're accepted.

**Schema Contracts:**

Every plugin declares its input and output schema:

```python
class MyTransform(RowTransform):
    name = "my_transform"

    input_schema = {
        "temperature": float,
        "humidity": float,
    }

    output_schema = {
        "temperature": float,
        "humidity": float,
        "heat_index": float,  # Added by this transform
    }
```

**Why Schemas Matter:**

| Benefit | Description |
|---------|-------------|
| **Pipeline validation** | Catch "plugin A outputs X, plugin B expects Y" at config time |
| **Coalesce configuration** | Must know how to merge N different outputs into 1 row |
| **Documentation** | Users know what each plugin expects/produces |
| **Landscape context** | Record "this was the expected shape" alongside actual data |

**Coalesce Schema Example:**

```yaml
plugins:
  - plugin: coalesce
    inputs:
      - path: stats_aggregation
        schema: { mean: float, max: float }
      - path: classifier
        schema: { category: string, confidence: float }
    output_schema:
      mean: float
      max: float
      category: string
      confidence: float
```

**Key Principle:** Plugins are stateless between rows (row transforms) OR explicitly stateful with clear flush semantics (aggregations). All plugins declare input/output schemas.

**Depends on:** Configuration (for plugin config and schema validation)

---

### 3. SDA Engine

**Intent:** The execution core. Orchestrates the Sense/Decide/Act flow.

**Responsibilities:**

- Load rows from source
- Process rows through transform chain
- Handle routing decisions (continue vs route to sink)
- Manage aggregation plugin state and flush
- Execute retry logic with attempt tracking
- Deliver rows to sinks
- Coordinate with Landscape for audit recording

**Components:**

| Component | Role |
|-----------|------|
| **RowProcessor** | Process single row through transforms, manage spans |
| **Orchestrator** | Coordinate source → transforms → sinks flow |
| **RetryManager** | Backoff, attempt tracking, quarantine decisions |
| **ArtifactPipeline** | Topological sort of sink dependencies |

**Row Lifecycle:**

```
Source.load()
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ For each row:                                                │
│                                                              │
│   Landscape.begin_row()                                      │
│       │                                                      │
│       ▼                                                      │
│   For each transform:                                        │
│       │                                                      │
│       ├─ [Row Transform/Gate]                                │
│       │      Landscape.begin_transform()                     │
│       │      transform.process(row) or gate.evaluate(row)    │
│       │      Landscape.complete_transform()                  │
│       │      If gate → route decision recorded               │
│       │                                                      │
│       └─ [Aggregation Transform/Gate]                        │
│              aggregation.accept(row)                         │
│              If trigger or flush:                            │
│                  Landscape.begin_transform()                 │
│                  aggregation.flush()                         │
│                  Landscape.complete_transform()              │
│       │                                                      │
│       ▼                                                      │
│   Route to sink (or continue to next transform)              │
│       │                                                      │
│       ▼                                                      │
│   Sink.write(row)                                            │
│   Landscape.register_artifact()                              │
└─────────────────────────────────────────────────────────────┘
```

**Aggregation in the Flow:**

Aggregation plugins complicate the simple row-by-row model:

1. **Accept phase**: Rows are offered to aggregation; may be buffered
2. **Trigger phase**: When threshold/signal/timeout, batch is processed
3. **Flush phase**: At end of source, all pending aggregations flush

The engine must track which rows are "in flight" in aggregations for audit purposes.

**Key Principle:** Reliability over performance. Explicit failure states, no silent drops.

**Depends on:** Plugin System, Landscape, Configuration

---

### 4. Configuration

**Intent:** Unified configuration with clear precedence and validation.

**Responsibilities:**

- Load configuration from multiple sources
- Apply precedence rules (system < pack < profile < suite < runtime)
- Interpolate environment variables
- Validate against Pydantic schemas
- Resolve plugin configurations

**Precedence Stack:**

```
┌─────────────────────────┐
│   Runtime overrides     │  ← Highest (CLI flags, env vars)
├─────────────────────────┤
│   Suite configuration   │  ← suite.yaml
├─────────────────────────┤
│   Profile configuration │  ← profiles/production.yaml
├─────────────────────────┤
│   Plugin pack defaults  │  ← packs/llm/defaults.yaml
├─────────────────────────┤
│   System defaults       │  ← Lowest (built-in)
└─────────────────────────┘
```

**Key Principle:** Resolved configuration is stored with each run. No "what was the config?" ambiguity.

**Depends on:** Nothing (foundational)

---

### 5. Payload Store

**Intent:** Separate large blobs from audit tables. Enable retention policies.

**Responsibilities:**

- Store large payloads (LLM responses, row data, artifacts)
- Return stable references for Landscape tables
- Apply retention and purge policies
- Optional compression
- Multiple backends (filesystem, S3, inline)

**Interface:**

```python
class PayloadStore(Protocol):
    def put(self, data: bytes, content_type: str) -> PayloadRef: ...
    def get(self, ref: PayloadRef) -> bytes | None: ...
    def exists(self, ref: PayloadRef) -> bool: ...
    def delete(self, ref: PayloadRef) -> bool: ...
```

**Retention Model:**

| Data Type | Default | After Expiry |
|-----------|---------|--------------|
| Row payloads | 90 days | Purge blob, keep hash in Landscape |
| Call payloads | 90 days | Purge blob, keep hash in Landscape |
| Artifacts | Per-policy | Per-policy |

**Key Principle:** Landscape keeps hashes forever; payloads are ephemeral. `explain()` degrades gracefully when payloads are purged.

**Depends on:** Canonical (for content hashing)

---

### 6. Canonical

**Intent:** Deterministic serialization for reliable hashing.

**Responsibilities:**

- Normalize pandas/numpy types to JSON primitives
- Produce deterministic JSON per RFC 8785 (JSON Canonicalization Scheme)
- Reject non-finite floats (NaN, Infinity)
- Version the canonicalization rules
- Compute stable SHA-256 hashes

**Two-Phase Approach:**

```
Phase 1: Normalize (our code)   Phase 2: Serialize (rfc8785)
─────────────────────────────   ────────────────────────────
numpy.int64 → int               rfc8785.dumps(normalized)
numpy.float64 → float
pd.Timestamp → UTC ISO          RFC 8785/JCS standard:
NaT/NA → None                   - Deterministic key order
NaN → ValueError                - Deterministic number format
                                - No whitespace
```

**Key Principle:** Hash algorithm is versioned (`sha256-rfc8785-v1`). Old runs remain verifiable under their recorded version.

**Depends on:** Nothing (foundational)

---

### 7. CLI

**Intent:** User-facing command interface.

**Responsibilities:**

- `elspeth run` - Execute a pipeline
- `elspeth explain` - Query lineage for a row/field
- `elspeth validate` - Check configuration without running
- `elspeth plugins` - List available plugins
- `elspeth status` - Check run status

**Key Principle:** Human-readable output by default, machine-readable (JSON) with `--json` flag.

**Depends on:** All other subsystems

---

## Subsystem Dependencies

```
                    ┌─────────┐
                    │   CLI   │
                    └────┬────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    SDA ENGINE                            │
└───────┬─────────────────┬─────────────────┬─────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Plugin System │ │   Landscape   │ │ Configuration │
└───────┬───────┘ └───────┬───────┘ └───────────────┘
        │                 │
        │                 ▼
        │         ┌───────────────┐
        │         │ Payload Store │
        │         └───────┬───────┘
        │                 │
        └────────┬────────┘
                 ▼
          ┌───────────────┐
          │   Canonical   │
          └───────────────┘
```

**Foundational (no dependencies):**

- Canonical
- Configuration

**Infrastructure (depend on foundational):**

- Payload Store → Canonical
- Plugin System → Configuration

**Core (depend on infrastructure):**

- Landscape → Canonical, Payload Store
- SDA Engine → Plugin System, Landscape, Configuration

**User-facing (depend on core):**

- CLI → Everything

---

## Design Document Status

| Subsystem | Document | Status |
|-----------|----------|--------|
| Overview | `subsystems/00-overview.md` | ✅ This document |
| Landscape | `subsystems/01-landscape.md` | 📝 To write |
| Plugin System | `subsystems/02-plugin-system.md` | 📝 To write |
| SDA Engine | `subsystems/03-sda-engine.md` | 📝 To write |
| Configuration | `subsystems/04-configuration.md` | 📝 To write |
| Payload Store | `subsystems/05-payload-store.md` | 📝 To write |
| Token Lifecycle | `subsystems/06-token-lifecycle.md` | ✅ Complete |
| Canonical | `architecture.md` | ✅ Complete |
| CLI | `subsystems/07-cli.md` | 📝 To write (low priority) |

---

## Aggregation Lineage Model

Aggregation plugins are **intentionally lossy** at the row level. When computing statistics or batch summaries, per-row detail is discarded - that's the point.

**What Landscape records for aggregations:**

| Level | What's Captured |
|-------|-----------------|
| Input | Which `row_id`s were accepted into the aggregation |
| Config | Aggregation settings (threshold, group keys, etc.) |
| Output | The aggregate result (stats, summary, batch decision) |

**What's NOT captured** (by design):

- Per-row contribution to the aggregate
- "Row 47 caused the average to be X"

This is the correct behavior. If you need row-level attribution, use a row transform, not an aggregation.

---

## Resolved Questions

| Question | Resolution |
|----------|------------|
| Partial flush failure | Batch membership persisted **before** processing. Failed batches retain members for retry/quarantine. |
| Aggregation lineage | Batch entity (`batches`, `batch_members`, `batch_outputs` tables) |
| Multi-aggregation chaining | Not allowed - aggregation collapses N→1, nothing left to aggregate |
| Multi-destination lineage | One `routing_event` per edge, linked by `routing_group_id` |
| Coalesce failure modes | Declared policy: `require_all`, `quorum(n)`, or `best_effort` |
| Fork/join identity | `token_id` + `token_parents` table for multi-parent joins |
| Coalesce correlation | Key strategy: `singleton`, `parent_token_id`, `row_id`, or `field(name)` |
| Static graph validation | `edges` table enables acyclicity check and "what could happen" queries |
| explain() in DAG | Use `token_id` for precision, or `row_id + sink` for disambiguation |
| Failed external calls | `calls.status` + nullable `response_hash` allows recording attempts without response |
| **Aggregation checkpointing** | Draft batch pattern: create batch on first `accept()`, persist members immediately. Crash recovery via query, not deserialization. No custom checkpoint logic needed. |

## Technology Decisions (Adopted)

**Design principle:** Prove the DAG infrastructure with deterministic transforms before adding external calls. A previous design iteration failed because LLMs were too tightly coupled to the orchestrator.

### Acceleration Stack

These libraries replace components that would become "mini-products" requiring ongoing maintenance:

| Component | Technology | Status | Rationale |
|-----------|------------|--------|-----------|
| **Canonical JSON** | `rfc8785` | ✅ ADOPTED | RFC 8785/JCS standard; we keep our normalization layer |
| **DAG Validation** | NetworkX | ✅ ADOPTED | Acyclicity checks, topological sort, graph queries |
| **Observability** | OpenTelemetry | ✅ ADOPTED | Emit spans from same events as Landscape; OTEL is view, Landscape is truth |
| **Tracing UI** | Jaeger | ✅ ADOPTED | Immediate visualization while building Landscape UI |
| **TUI** | Textual | ✅ ADOPTED | Interactive terminal UI for `explain`, `status` |
| **Logging** | structlog | ✅ ADOPTED | Structured key/value events alongside Landscape |
| **Rate Limiting** | pyrate-limiter | ✅ ADOPTED | Multi-limit/interval support, SQLite/Redis persistence |
| **Diffing** | DeepDiff | ✅ ADOPTED | Deep nested diffs for verify mode (recorded vs live) |
| **Property Testing** | Hypothesis | ✅ ADOPTED | Find nasty canonicalization, DAG, and lineage bugs |

### Deferred to Phase 6+

| Component | Technology | Status | Rationale |
|-----------|------------|--------|-----------|
| **LLM Plugin Pack** | LiteLLM | ⏳ PHASE 6 | Prove DAG infrastructure first |
| **Azure Plugin Pack** | azure-storage-blob | ⏳ PHASE 7 | Cloud integration after core is stable |

## Open Questions

1. **Unsupported type handling**: Should `numpy.datetime64` / `numpy.timedelta64` fail loudly with a clear message, or silently delegate to `json.dumps` TypeError?
