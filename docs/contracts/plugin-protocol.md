# ELSPETH Plugin Protocol Contract

> **Status:** FINAL (v1.1)
> **Last Updated:** 2026-01-17
> **Authority:** This document is the master reference for all plugin interactions.

## Overview

ELSPETH follows the Sense/Decide/Act (SDA) model:

```
SOURCE (Sense) → DECIDE → SINK (Act)
```

**Critical distinction:**

| Layer | What | Who Writes It |
|-------|------|---------------|
| **User Plugins** | External system integration + business logic | Plugin authors |
| **System Operations** | Token routing, batching, forking, merging | ELSPETH (config-driven) |

### User Plugins (This Document)

These are **code** that plugin authors write:

| Plugin | Purpose | Touches |
|--------|---------|---------|
| **Source** | Load data from external systems | Row contents |
| **Transform** | Apply business logic to rows | Row contents |
| **Sink** | Write data to external systems | Row contents |

### System Operations (NOT Plugins)

These are **config-driven** infrastructure provided by ELSPETH:

| Operation | Purpose | Config Example |
|-----------|---------|----------------|
| **Gate** | Route tokens based on conditions | `condition: "row['score'] > 0.8"` |
| **Aggregation** | Batch tokens until trigger | `trigger: "count >= 100"` |
| **Fork** | Split token to parallel paths | Routing action |
| **Coalesce** | Merge tokens from parallel paths | `policy: require_all` |

System operations work on **wrapped data** (tokens, metadata) and are 100% ELSPETH code.
User plugins work on **row contents** (the actual data) and are plugin author code.

---

## Core Principles

### 1. Audit Is Non-Negotiable

Every plugin interaction is recorded. The audit trail must answer:
- What data entered the plugin?
- What did the plugin produce?
- When did it happen?
- Did it succeed or fail?

Plugins MUST return audit-relevant information. "Trust me, I did it" is not acceptable.

### 2. Plugins Control Their Own Schedule

ELSPETH doesn't dictate internal timing. Plugins may:
- Block waiting for external resources (satellite links, APIs, databases)
- Queue work internally and process on their own schedule
- Batch operations for efficiency

The contract specifies WHEN methods are called, not HOW FAST plugins must respond.

### 3. Exception Handling: Their Data vs Our Code

This mirrors ELSPETH's core Data Manifesto: **Their Data** (user input) gets tolerance, **Our Data/Code** gets zero tolerance.

#### The Divide-By-Zero Test

```python
# THEIR DATA caused the error → TOLERATE
def process(self, row, ctx):
    try:
        result = row["value"] / row["divisor"]  # User passed divisor=0
    except ZeroDivisionError:
        return TransformResult.error({"reason": "division_by_zero", "field": "divisor"})
    return TransformResult.success({"result": result})

# OUR CODE caused the error → CRASH
def process(self, row, ctx):
    # If _batch_count is 0, that's MY bug - I should have initialized it
    average = self._total / self._batch_count  # Let it crash!
    return TransformResult.success({"average": average})
```

#### The Boundary

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PLUGIN PROCESSING                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  THEIR DATA (row fields, external input)                            │
│  ─────────────────────────────────────────                          │
│  • Wrap operations in try/catch                                      │
│  • Return error result on failure                                    │
│  • Row gets quarantined, pipeline continues                          │
│                                                                      │
│  Examples:                                                           │
│  • row["field"] doesn't exist → catch KeyError, return error        │
│  • row["value"] / row["divisor"] → catch ZeroDivisionError          │
│  • int(row["count"]) → catch ValueError                             │
│  • external_api.call(row["id"]) → catch ApiError                    │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  OUR CODE (internal state, plugin logic)                            │
│  ─────────────────────────────────────────                          │
│  • Do NOT wrap in try/catch                                          │
│  • Let exceptions propagate                                          │
│  • Pipeline crashes - this is a bug to fix                           │
│                                                                      │
│  Examples:                                                           │
│  • self._counter / self._batch_size → if batch_size=0, that's a bug │
│  • self._buffer[index] → if index wrong, that's a bug               │
│  • self._connection.execute() → if connection is None, that's a bug │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Contract

| Zone | On Error | Plugin Action | ELSPETH Action |
|------|----------|---------------|----------------|
| **Their Data** | User row causes error | Catch, return error result | Quarantine row, continue |
| **Our Code** | Internal state causes error | Let it crash | Record in audit, halt pipeline |
| **Lifecycle** | `on_start`/`on_complete`/`close` fails | Let it crash | Halt pipeline (config/code bug) |

**Rules:**
- Plugins MUST wrap operations on user-provided data and return error results
- Plugins MUST NOT wrap operations on internal state - let bugs surface
- If you're tempted to add `try/except` around your own logic, you have a bug to fix

### 4. Forced Pass for Lifecycle Hooks

All lifecycle hooks are REQUIRED in the protocol, even if implementation is `pass`.

**Why:** The audit trail records every lifecycle event. Even an empty `on_start()` produces an audit record: "plugin started at timestamp X". This is defensible under audit.

---

## Plugin Types

### Source

**Purpose:** Load data into the pipeline. Exactly one source per run.

**Data Flow:** Produces rows on its own schedule.

#### Required Attributes

```python
name: str                          # Plugin identifier (e.g., "csv", "api", "satellite")
output_schema: type[PluginSchema]  # Schema of rows this source produces
node_id: str | None                # Set by orchestrator after registration
determinism: Determinism           # DETERMINISTIC, NON_DETERMINISTIC, or EXTERNAL
plugin_version: str                # Semantic version for reproducibility
```

#### Required Methods

```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize with configuration.

    Called once at pipeline construction.
    Validate config here - fail fast if misconfigured.
    """

def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
    """Yield rows from the source.

    MAY BLOCK internally waiting for data (satellite downlink, API polling).
    Yields rows on source's own schedule.
    Iterator exhausts when source has no more data.

    Returns:
        Iterator yielding row dicts matching output_schema
    """

def close(self) -> None:
    """Release resources.

    Called after on_complete() or on error.
    MUST NOT raise - log errors internally if cleanup fails.
    """
```

#### Required Lifecycle Hooks

```python
def on_start(self, ctx: PluginContext) -> None:
    """Called before load().

    Use for: Opening connections, authenticating, preparing resources.
    Should be reasonably quick - heavy blocking belongs in load().

    If this fails, it's a CODE BUG - pipeline crashes.
    """

def on_complete(self, ctx: PluginContext) -> None:
    """Called after load() exhausts (before close).

    Use for: Finalization, recording completion metrics.

    If this fails, it's a CODE BUG - pipeline crashes.
    """
```

#### Lifecycle

```
__init__(config)
    │
    ▼
on_start(ctx)           ← Setup connections, auth
    │
    ▼
load(ctx) ─► yields rows on source's schedule
    │        (may block internally)
    │
    ▼ (iterator exhausts)
on_complete(ctx)        ← Finalization
    │
    ▼
close()                 ← Release resources
```

#### Audit Records

- `on_start` called timestamp
- Each row yielded (row_id, content_hash)
- `on_complete` called timestamp
- Total rows produced
- Any errors encountered

---

### Transform

**Purpose:** Apply business logic to a row. Stateless between rows.

**Data Flow:** One row in → one row out (possibly modified).

#### Required Attributes

```python
name: str
input_schema: type[PluginSchema]
output_schema: type[PluginSchema]
node_id: str | None
determinism: Determinism
plugin_version: str
```

#### Required Methods

```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize with configuration."""

def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
    """Process a single row.

    MUST be a pure function for DETERMINISTIC transforms:
    - Same input → same output
    - No side effects

    Returns:
        TransformResult.success(row) - processed row
        TransformResult.error(reason, retryable=bool) - processing failed

    Exception handling:
        - Data validation errors → return TransformResult.error()
        - Code bugs → let exception propagate (will crash)
    """

def close(self) -> None:
    """Release resources."""
```

#### Required Lifecycle Hooks

```python
def on_start(self, ctx: PluginContext) -> None:
    """Called before any rows are processed."""

def on_complete(self, ctx: PluginContext) -> None:
    """Called after all rows are processed."""
```

#### TransformResult Contract

```python
@dataclass
class TransformResult:
    status: Literal["success", "error"]
    row: dict[str, Any] | None      # Output row (success) or None (error)
    reason: dict[str, Any] | None   # Error details or None (success)
    retryable: bool = False         # Can this operation be retried?

    # Audit fields (set by executor, NOT by plugin)
    input_hash: str | None
    output_hash: str | None
    duration_ms: float | None
```

**Factory methods:**

```python
TransformResult.success(row)                    # Success with output
TransformResult.error(reason, retryable=False)  # Failure
```

#### Lifecycle

```
__init__(config)
    │
    ▼
on_start(ctx)
    │
    ▼
┌─────────────────────────────┐
│ process(row, ctx) × N       │  ← Called for each row
│     │                       │
│     ▼                       │
│ TransformResult             │
└─────────────────────────────┘
    │
    ▼
on_complete(ctx)
    │
    ▼
close()
```

#### Audit Records

- Each `process()` call: input_hash, output_hash, duration_ms, status
- Errors: exception type, message, retryable flag

---

### Sink

**Purpose:** Output data to external destination. One or more per run.

**Data Flow:** Receives rows, produces artifacts.

#### Required Attributes

```python
name: str
input_schema: type[PluginSchema]
node_id: str | None
idempotent: bool                # Can this sink handle duplicate writes safely?
determinism: Determinism
plugin_version: str
```

#### Required Methods

```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize with configuration."""

def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
    """Receive rows and return proof of work.

    The sink controls its own internal processing:
    - May write immediately (CSV, database)
    - May queue internally and process later (satellite, async API)
    - May batch for efficiency

    MUST return ArtifactDescriptor describing what was produced/queued.
    SHOULD NOT block for slow operations - queue internally, confirm in on_complete().

    Returns:
        ArtifactDescriptor with content_hash and size_bytes (REQUIRED for audit)
    """

def flush(self) -> None:
    """Flush any buffered data.

    Called periodically and before on_complete().
    """

def close(self) -> None:
    """Release resources.

    Called after on_complete() or on error.
    """
```

#### Required Lifecycle Hooks

```python
def on_start(self, ctx: PluginContext) -> None:
    """Called before any writes.

    Use for: Opening connections, creating output files, initializing queues.
    """

def on_complete(self, ctx: PluginContext) -> None:
    """Called after all writes, before close.

    CRITICAL: This method MAY BLOCK until all queued work is confirmed.

    Use for:
    - Committing database transactions
    - Waiting for satellite transmission confirmation
    - Finalizing multi-part uploads
    - Any operation that must complete before run is considered done

    The run CANNOT complete until on_complete() returns for ALL sinks.
    """
```

#### ArtifactDescriptor Contract

```python
@dataclass(frozen=True)
class ArtifactDescriptor:
    artifact_type: Literal["file", "database", "webhook"]
    path_or_uri: str              # Where the artifact lives
    content_hash: str             # SHA-256 of content (REQUIRED)
    size_bytes: int               # Size of artifact (REQUIRED)
    metadata: dict | None = None  # Type-specific metadata
```

**Factory methods:**

```python
ArtifactDescriptor.for_file(path, content_hash, size_bytes)
ArtifactDescriptor.for_database(url, table, content_hash, payload_size, row_count)
ArtifactDescriptor.for_webhook(url, content_hash, request_size, response_code)
```

#### content_hash Requirement

The `content_hash` field is REQUIRED and proves what was written. Hash computation differs by artifact type:

| Artifact Type | What Gets Hashed | Why |
|---------------|------------------|-----|
| **file** | SHA-256 of file contents | Proves exact bytes written |
| **database** | SHA-256 of canonical JSON payload BEFORE insert | Proves intent (DB may transform data) |
| **webhook** | SHA-256 of request body | Proves what was sent (response is separate) |

**Key principle:** Hash what YOU control, not what the destination does with it. For databases, you hash the payload you're sending, not what the DB stores (it may add timestamps, auto-increment IDs, etc.). This proves intent.

#### Lifecycle

```
__init__(config)
    │
    ▼
on_start(ctx)               ← Open connections, prepare
    │
    ▼
┌──────────────────────────────────────────────┐
│ write(rows, ctx) → ArtifactDescriptor        │  ← May be called multiple times
│     │                                        │
│     ▼                                        │
│ (sink processes on its own schedule)         │
└──────────────────────────────────────────────┘
    │
    ▼
flush()                     ← Flush any buffers
    │
    ▼
on_complete(ctx)            ← BLOCKS until truly done
    │                         (satellite confirms, transaction commits)
    ▼
close()                     ← Release resources
```

#### Idempotency

Sinks declare `idempotent: bool`:
- `True`: Safe to retry writes with same data (uses idempotency key)
- `False`: Retry may cause duplicates (append-only files, non-idempotent APIs)

Idempotency key format: `{run_id}:{token_id}:{sink_name}`

#### Audit Records

- Each `write()`: ArtifactDescriptor (type, path, hash, size)
- `on_complete` timestamp (confirms delivery)
- Idempotency key if applicable

---

## System Operations (Engine-Level)

The following are **engine-level operations** with config-driven behavior. They are NOT user-written plugins. They operate on **wrapped data** (tokens, routing metadata) rather than row contents.

**Why these are engine-level, not plugins:**
- They don't touch row contents - they manipulate token flow
- They require coordination across the DAG (fork/join semantics)
- Their behavior is fully expressible via configuration
- They are 100% ELSPETH code with no user extension points

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA FLOW ARCHITECTURE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  USER PLUGINS (touch row contents)                               │
│  ─────────────────────────────────                               │
│  Source ──► [row data] ──► Transform ──► [row data] ──► Sink    │
│                                                                   │
│  SYSTEM OPERATIONS (touch token wrapper)                         │
│  ───────────────────────────────────────                         │
│  Gate: "where does this token go?"                               │
│  Fork: "copy token to parallel paths"                            │
│  Coalesce: "merge tokens from paths"                             │
│  Aggregation: "batch tokens until trigger"                       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### Gate (Routing Decision)

**Purpose:** Evaluate a condition on row data and decide where the token goes next.

**Key property:** Gates don't modify row data - they only make routing decisions.

#### Configuration

```yaml
pipeline:
  - source: csv_input

  - gate: quality_check
    condition: "row['confidence'] >= 0.85"
    routes:
      high: continue          # Continue to next node
      low: review_sink        # Route to named sink

  - transform: enrich_data

  - sink: output
```

#### How It Works

1. Engine evaluates `condition` expression against row data
2. Expression returns a route label (`high`, `low`, etc.)
3. Engine looks up route label in `routes` config
4. Token is routed to destination (`continue` or sink name)

#### Expression Language

```python
# Simple field comparison
"row['score'] > 0.8"

# Multiple conditions
"row['status'] == 'active' and row['balance'] > 0"

# Field existence check
"'optional_field' in row"

# Null handling
"row.get('nullable_field') is not None"
```

#### Expression Safety

Gate conditions are evaluated using a **restricted expression parser**, NOT Python `eval()`.

**Allowed:**
- Field access: `row['field']`, `row.get('field')`
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Boolean operators: `and`, `or`, `not`
- Membership: `in`, `not in`
- Literals: strings, numbers, booleans, None
- List/dict literals for membership checks

**NOT Allowed:**
- Function calls (except `row.get()`)
- Imports
- Attribute access beyond row fields
- Assignment
- Lambda/comprehensions

This prevents code injection. An expression like `"__import__('os').system('rm -rf /')"` will be rejected at config validation time.

#### Audit Trail

- Condition evaluated: expression + result
- Route chosen: label + destination
- Timing: evaluation duration

---

### Fork (Token Splitting)

**Purpose:** Copy a single token to multiple parallel paths for concurrent processing.

**Key property:** Creates child tokens with same row data, different branch identities.

#### Configuration

```yaml
pipeline:
  - source: input

  - gate: parallel_analysis
    condition: "True"  # Always fork
    routes:
      all: fork         # Special keyword triggers fork
    fork_to:
      - sentiment_path
      - entity_path
      - summary_path

  # Parallel paths defined separately
  paths:
    sentiment_path:
      - transform: sentiment_analyzer
    entity_path:
      - transform: entity_extractor
    summary_path:
      - transform: summarizer
```

#### How It Works

1. Gate evaluates condition
2. If route is `fork`, engine creates N child tokens
3. Each child token:
   - Has same `row_id` as parent
   - Gets unique `token_id`
   - Records `parent_token_id` for lineage
   - Assigned to specific path/branch
4. Parent token marked with terminal state `FORKED`
5. Child tokens flow through their respective paths

#### Token Lineage

```
Parent Token (T1)
    │ row_id: R1
    │ status: FORKED
    │
    ├──► Child Token (T2)
    │    row_id: R1
    │    parent_token_id: T1
    │    branch: sentiment_path
    │
    ├──► Child Token (T3)
    │    row_id: R1
    │    parent_token_id: T1
    │    branch: entity_path
    │
    └──► Child Token (T4)
         row_id: R1
         parent_token_id: T1
         branch: summary_path
```

#### Audit Trail

- Fork event: parent_token_id, child_token_ids, branch assignments
- Fork group ID: links all children from same fork

---

### Coalesce (Token Merging)

**Purpose:** Merge tokens from parallel paths back into a single token.

**Key property:** Waits for tokens from specified branches, combines based on policy.

#### Configuration

```yaml
pipeline:
  # ... fork happens earlier ...

  - coalesce: merge_results
    branches:
      - sentiment_path
      - entity_path
      - summary_path
    policy: require_all    # Wait for all branches
    timeout: 5m            # Max wait time
    merge: union           # How to combine row data
```

#### Policies

| Policy | Behavior |
|--------|----------|
| `require_all` | Wait for ALL branches, fail if any missing after timeout |
| `quorum` | Wait for N branches (configurable threshold) |
| `best_effort` | Wait until timeout, merge whatever arrived |
| `first` | Take first arrival, discard others |

#### Merge Strategies

| Strategy | Behavior |
|----------|----------|
| `union` | Combine all fields (later branches overwrite earlier) |
| `nested` | Each branch output becomes nested object: `{branch_name: output}` |
| `select` | Take output from specific branch only |

#### How It Works

1. Engine tracks arriving tokens by `row_id` and branch
2. When policy is satisfied (all arrived, quorum met, timeout):
   - Creates new merged token
   - Combines row data per merge strategy
   - Child tokens marked with terminal state `COALESCED`
3. Merged token continues down pipeline

#### Token Lineage

```
Child Tokens (arriving)
    │
    ├── T2 (sentiment_path): {sentiment: "positive"}
    ├── T3 (entity_path): {entities: ["ACME", "NYC"]}
    └── T4 (summary_path): {summary: "..."}
    │
    ▼ (coalesce with union strategy)
    │
Merged Token (T5)
    row_id: R1
    parent_token_ids: [T2, T3, T4]
    row_data: {
        sentiment: "positive",
        entities: ["ACME", "NYC"],
        summary: "..."
    }
```

#### Audit Trail

- Coalesce event: input_token_ids, output_token_id, policy used
- Timing: wait duration, which branches arrived when
- Merge details: strategy used, any conflicts

---

### Aggregation (Token Batching)

**Purpose:** Collect multiple tokens until a trigger fires, then release as batch.

**Key property:** Converts N input tokens into M output tokens (often N→1 for aggregates).

#### Configuration

```yaml
pipeline:
  - source: events

  - aggregate: batch_hourly
    trigger:
      count: 1000           # Fire after 1000 rows
      timeout: 1h           # Or after 1 hour
      condition: "row['type'] == 'flush_signal'"  # Or on special row
    output: single          # Emit one summary row (vs 'passthrough')
```

#### Trigger Types

| Trigger | Fires When |
|---------|------------|
| `count` | N tokens accumulated |
| `timeout` | Duration elapsed since batch start |
| `condition` | Row matches expression |
| `end_of_source` | Source exhausted (implicit, always checked) |

Multiple triggers can be combined (first one to fire wins).

#### Output Modes

| Mode | Behavior |
|------|----------|
| `single` | Batch produces one output token (aggregated result) |
| `passthrough` | Batch releases all accumulated tokens |
| `transform` | Apply transform to batch, emit result(s) |

#### How It Works

1. Tokens arrive at aggregation node
2. Engine adds token to current batch:
   - Assigns `batch_id`
   - Records batch membership in audit trail
   - Token marked `CONSUMED_IN_BATCH`
3. Engine checks trigger conditions
4. When trigger fires:
   - Batch state: `draft` → `executing` → `completed`
   - Output token(s) created and continue downstream
   - Batch membership recorded for audit

#### Batch Lifecycle

```
Batch B1 (draft)
    │
    ├── Accept T1 → batch_members: [T1]
    ├── Accept T2 → batch_members: [T1, T2]
    ├── Accept T3 → batch_members: [T1, T2, T3]
    │
    ▼ (trigger: count >= 3)
    │
Batch B1 (executing)
    │
    ▼ (compute aggregate)
    │
Batch B1 (completed)
    │
    └──► Output Token T4
         row_data: {count: 3, sum: 150, avg: 50}

Input tokens T1, T2, T3 → terminal state: CONSUMED_IN_BATCH
```

#### Audit Trail

- Batch created: batch_id, trigger config
- Batch membership: which tokens belong to which batch
- Batch state transitions: draft → executing → completed/failed
- Trigger event: which condition fired, when

---

### Why Not Plugins?

These operations are engine-level because:

1. **They don't touch row contents** - A gate evaluates a condition but doesn't modify the row. Fork copies tokens but doesn't change data. Coalesce combines outputs but the merge strategy is config, not code.

2. **They require DAG coordination** - Fork/coalesce semantics span multiple nodes. The engine must track token lineage, manage parallel paths, handle timeouts. This is orchestration, not business logic.

3. **Config is sufficient** - All behavior is expressible via:
   - Expressions: `"row['field'] > value"`
   - Policies: `require_all`, `quorum`, `best_effort`
   - Strategies: `union`, `nested`, `select`

4. **100% our code** - These are ELSPETH internals with comprehensive testing. No user extension points, no defensive programming needed.

---

## Exception Handling Summary

| Method | On Failure |
|--------|------------|
| `__init__` | Raise immediately - misconfiguration |
| `on_start` | CODE BUG → crash |
| `on_complete` | CODE BUG → crash |
| `close` | Log error, don't raise |
| `process` (Transform) | Data error → return error result; Code bug → raise |
| `write` (Sink) | Data error → return error result; Code bug → raise |
| `load` (Source) | May raise StopIteration (normal), other errors → crash |

---

## Determinism Declaration

Plugins declare their determinism level:

```python
class Determinism(Enum):
    DETERMINISTIC = "deterministic"          # Same input → same output, always
    NON_DETERMINISTIC = "non_deterministic"  # Output may vary (timestamps, random)
    EXTERNAL = "external"                    # Depends on external state (API, DB)
```

**Implications:**
- `DETERMINISTIC`: Safe to replay, verify mode can compare outputs
- `NON_DETERMINISTIC`: Replay produces different output (expected)
- `EXTERNAL`: Output depends on external world state

---

## Engine Concerns (Not Plugin Contract)

The following are handled by the ELSPETH engine, not by plugins. They are documented here for completeness but are NOT part of the plugin contract.

### Schema Validation

The engine validates schema compatibility between connected nodes at pipeline construction time:
- Source `output_schema` → Transform `input_schema`
- Transform `output_schema` → next Transform `input_schema` or Sink `input_schema`

Plugins declare schemas; the engine enforces compatibility. Plugins do NOT validate schemas themselves.

### Retry Policy

Retry behavior is configured at the engine level, not declared by plugins:

```yaml
retry:
  max_attempts: 3
  base_delay: 1.0
  max_delay: 60.0
  jitter: 1.0
```

Plugins indicate whether errors are `retryable` in their result objects. The engine decides whether and when to retry based on policy.

### Rate Limiting

External call rate limits are engine configuration:

```yaml
rate_limit:
  calls_per_second: 10
  burst: 20
```

Plugins make calls; the engine throttles them.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | 2026-01-17 | Add content_hash requirements, expression safety, engine concerns |
| 1.0 | 2026-01-17 | Initial contract - Source, Transform, Sink + System Operations |
