# ELSPETH Plugin Protocol Contract

> **Status:** FINAL (v1.6)
> **Last Updated:** 2026-01-20
> **Authority:** This document is the master reference for all plugin interactions.

> ⚠️ **Implementation Status:** Features in v1.5 (multi-row output, `creates_tokens`, aggregation output modes) are **specified but not yet implemented**. See [`docs/plans/2026-01-19-multi-row-output.md`](../plans/2026-01-19-multi-row-output.md) for the implementation plan.

## Overview

ELSPETH follows the Sense/Decide/Act (SDA) model:

```
SOURCE (Sense) → DECIDE → SINK (Act)
```

**Critical distinction:**

| Layer | What | Requires Code? |
|-------|------|----------------|
| **Plugins** | External system integration + business logic | Yes (validated system code) |
| **System Operations** | Token routing, batching, forking, merging | No (config-driven) |

### Plugins (This Document)

Plugins are **system code** developed by the same team that operates ELSPETH. They are independently validated and treated as trusted components. This is NOT a plugin marketplace—arbitrary external code is not accepted.

| Plugin | Purpose | Touches |
|--------|---------|---------|
| **Source** | Load data from external systems | Row contents |
| **Transform** | Apply business logic to rows | Row contents |
| **Sink** | Write data to external systems | Row contents |

### System Operations (NOT Plugins)

These are **config-driven** infrastructure provided by the ELSPETH engine:

| Operation | Purpose | Config Example |
|-----------|---------|----------------|
| **Gate** | Route tokens based on conditions | `condition: "row['score'] > 0.8"` |
| **Aggregation** | Batch tokens until trigger | `trigger: "count >= 100"` |
| **Fork** | Split token to parallel paths | Routing action |
| **Coalesce** | Merge tokens from parallel paths | `policy: require_all` |

System operations work on **wrapped data** (tokens, metadata) and require no custom code.
Plugins work on **row contents** (the actual data) and are validated system code.

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

### 3. Exception Handling: Three-Tier Trust Model

ELSPETH uses a three-tier trust model that determines how exceptions should be handled:

| Tier | Trust Level | Coercion | Exception Handling |
|------|-------------|----------|-------------------|
| **External Data** (Source input) | Zero trust | ✅ Allowed | Validate, coerce, quarantine failures |
| **Pipeline Data** (Post-source rows) | Elevated ("probably ok") | ❌ Forbidden | Types trusted, wrap VALUE operations |
| **Our Code** (Plugin internals) | Full trust | ❌ N/A | Let it crash - bugs must surface |

#### The Key Insight: Type-Safe ≠ Operation-Safe

Data that passed source validation has correct **types**, but **values** can still cause operation failures:

```python
# Pipeline data is type-valid (int), but operation fails
row = {"divisor": 0}  # Passed source validation ✓
result = 100 / row["divisor"]  # 💥 ZeroDivisionError - WRAP THIS

# Pipeline data is type-valid (str), but content is problematic
row = {"date": "not-a-date"}  # Passed as str ✓
parsed = datetime.fromisoformat(row["date"])  # 💥 ValueError - WRAP THIS
```

#### The Divide-By-Zero Test

```python
# THEIR DATA value caused the error → WRAP AND HANDLE
def process(self, row, ctx):
    try:
        result = row["value"] / row["divisor"]  # User's divisor=0
    except ZeroDivisionError:
        return TransformResult.error({"reason": "division_by_zero", "field": "divisor"})
    return TransformResult.success({"result": result})

# OUR CODE caused the error → LET IT CRASH
def process(self, row, ctx):
    # If _batch_count is 0, that's MY bug - I should have initialized it
    average = self._total / self._batch_count  # Let it crash!
    return TransformResult.success({"average": average})
```

#### Coercion Rules by Plugin Type

| Plugin Type | May Coerce Types? | Why |
|-------------|-------------------|-----|
| **Source** | ✅ Yes | Normalizes external data at ingestion boundary (`"42"` → `42`) |
| **Transform** | ❌ No | Receives validated data; wrong types = upstream bug |
| **Sink** | ❌ No | Receives validated data; wrong types = upstream bug |

If a transform receives `"42"` when its `input_schema` says `int`, that's a bug in the source or upstream transform. The correct fix is to fix the upstream plugin, NOT to coerce in the transform.

#### The Boundary

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PLUGIN PROCESSING                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  THEIR DATA VALUES (row field values, external responses)           │
│  ─────────────────────────────────────────────────────────          │
│  • Types are trusted (source validated them)                        │
│  • Values may still cause operation failures                        │
│  • Wrap OPERATIONS in try/catch                                     │
│  • Return error result on failure                                    │
│  • Row gets quarantined, pipeline continues                          │
│                                                                      │
│  Examples:                                                           │
│  • row["value"] / row["divisor"] → catch ZeroDivisionError          │
│  • datetime.fromisoformat(row["date"]) → catch ValueError           │
│  • external_api.call(row["id"]) → catch ApiError                    │
│  • json.loads(row["payload"]) → catch JSONDecodeError               │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  THEIR DATA TYPES (at Source boundary ONLY)                         │
│  ─────────────────────────────────────────────────────────          │
│  • External data may have wrong types                               │
│  • Sources MAY coerce: "42" → 42, "true" → True                     │
│  • Sources MUST quarantine rows that can't be coerced               │
│  • Transforms/Sinks MUST NOT coerce types                           │
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
| **Their Data Values** | Row value causes operation error | Catch, return error result | Quarantine row, continue |
| **Their Data Types** | Wrong type at Source boundary | Coerce if possible, else error | Quarantine row, continue |
| **Their Data Types** | Wrong type at Transform/Sink | — | This is an upstream bug, should crash |
| **Our Code** | Internal state causes error | Let it crash | Record in audit, halt pipeline |
| **Lifecycle** | `on_start`/`on_complete`/`close` fails | Let it crash | Halt pipeline (config/code bug) |

**Rules:**
- Plugins MUST wrap operations on row values and return error results on failure
- Sources MAY coerce external data types; Transforms/Sinks MUST NOT
- Plugins MUST NOT wrap operations on internal state - let bugs surface
- If you're tempted to add `try/except` around your own logic, you have a bug to fix
- If a transform receives wrong types, that's an upstream bug to fix, not something to coerce

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

#### Required Configuration

```yaml
sources:
  my_source:
    type: csv_source
    path: data/input.csv
    schema:
      mode: strict
      fields:
        name: {type: string}
        age: {type: integer}
    # REQUIRED: Where do non-conformant rows go?
    on_validation_failure: quarantine_sink  # Sink name, or "discard"
```

**`on_validation_failure`** (REQUIRED):
- Specifies destination for rows that fail schema validation/coercion
- Value must be a sink name or `"discard"` for explicit drop
- Cannot be omitted, empty, or null - operator must acknowledge bad data handling
- Even when `"discard"`, a `QuarantineEvent` is recorded in the audit trail

**Quarantine behavior:**
1. Source attempts to validate/coerce row against `output_schema`
2. If validation fails:
   - `QuarantineEvent` recorded (always, even for discard)
   - Row routed to configured sink OR dropped if `"discard"`
   - Row does NOT enter the pipeline
3. If validation succeeds:
   - Row enters pipeline as normal

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
- **QuarantineEvent** for each validation failure:
  - `run_id`, `source_id`, `row_index`
  - `raw_row` (original data before coercion attempt)
  - `failure_reason` (why validation failed)
  - `field_errors` (per-field error details)
  - `destination` (sink name or "discard")
  - `timestamp`

---

### Transform

**Purpose:** Apply business logic to rows. Stateless between calls.

**Data Flow:** Row(s) in → row(s) out (possibly modified).

#### Required Attributes

```python
name: str
input_schema: type[PluginSchema]
output_schema: type[PluginSchema]
node_id: str | None
determinism: Determinism
plugin_version: str
is_batch_aware: bool = False  # Set True for batch processing at aggregation nodes
creates_tokens: bool = False  # Set True for deaggregation (1→N row expansion)
```

#### Token Creation (Deaggregation)

Transforms that expand one row into multiple rows must declare `creates_tokens = True`:

```python
class JSONExplode(BaseTransform):
    name = "json_explode"
    creates_tokens = True  # Engine creates new tokens for each output row
    # ...

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        items = row["items"]  # Trust: source validated this is a list
        return TransformResult.success_multi([
            {**row, "item": item, "item_index": i}
            for i, item in enumerate(items)
        ])
```

**Invariants:**
- `creates_tokens=True` + `success()` → single output (allowed, like passthrough)
- `creates_tokens=True` + `success_multi()` → engine creates new tokens per output row
- `creates_tokens=False` + `success_multi()` → RuntimeError (except in aggregation passthrough mode)
- `success_multi([])` is invalid (no silent drops) - use `success()` for a single-row empty case

**Engine semantics:**
- When a deaggregation transform (`creates_tokens=True`) returns `success_multi()` for a single input token, the engine creates N child tokens via `expand_token()` and the parent token reaches terminal state `EXPANDED` (children continue).

#### Batch-Aware Transforms

Transforms can declare `is_batch_aware = True` to receive batched rows when used at aggregation nodes:

```python
class SummaryTransform(BaseTransform):
    name = "summary"
    is_batch_aware = True  # Engine will pass list[dict] when used at aggregation node
    # ... other attrs ...

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        if isinstance(row, list):
            # Batch mode: aggregate the rows
            total = sum(r["value"] for r in row)
            return TransformResult.success({"total": total, "count": len(row)})
        # Single row mode
        return TransformResult.success(row)
```

**How it works:**

1. Transform is configured at an aggregation node (see "Aggregation" below)
2. Engine buffers rows until trigger fires (count, timeout, condition)
3. Engine calls `transform.process(rows: list[dict], ctx)` with the batch
4. Transform returns aggregated result

**Key points:**
- `is_batch_aware = False` (default): Transform always receives single `dict`
- `is_batch_aware = True`: Transform MAY receive `list[dict]` at aggregation nodes
- Transforms should handle both cases if `is_batch_aware = True`
- The engine decides when to batch based on pipeline configuration

#### Optional Configuration

```yaml
transforms:
  price_calculator:
    type: custom_transform
    # OPTIONAL: Where do rows go when transform returns error?
    on_error: failed_calculations  # Sink name, or "discard"
```

**`on_error`** (OPTIONAL):
- Specifies destination for rows where transform returns `TransformResult.error()`
- Value must be a sink name or `"discard"` for explicit drop
- If omitted and transform returns an error → `ConfigurationError` (pipeline crashes)
- Even when `"discard"`, a `TransformErrorEvent` is recorded in the audit trail

**Error vs Bug distinction:**

| Scenario | Signal | Behavior |
|----------|--------|----------|
| **Processing Error** | `TransformResult.error(...)` | Route to `on_error` sink |
| **Transform Bug** | Exception thrown | CRASH immediately |

```python
# PROCESSING ERROR - legitimate, uses on_error routing
def process(self, row: dict, ctx: PluginContext) -> TransformResult:
    if row["quantity"] == 0:
        return TransformResult.error({"reason": "division_by_zero"})
    return TransformResult.success({"unit_price": row["total"] / row["quantity"]})

# TRANSFORM BUG - crashes, does NOT use on_error routing
def process(self, row: dict, ctx: PluginContext) -> TransformResult:
    return TransformResult.success({"value": row["nonexistent"]})  # KeyError = BUG
```

#### Required Methods

```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize with configuration."""

def process(
    self,
    row: dict[str, Any] | list[dict[str, Any]],
    ctx: PluginContext,
) -> TransformResult:
    """Process row(s).

    For non-batch-aware transforms (is_batch_aware=False):
    - `row` is always a single dict
    - MUST be a pure function for DETERMINISTIC transforms

    For batch-aware transforms (is_batch_aware=True):
    - `row` may be a list[dict] when used at aggregation nodes
    - Should handle both single dict and list[dict] cases

    Returns:
        TransformResult.success(row) - processed row(s)
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
    row: dict[str, Any] | None           # Single output row (mutually exclusive with rows)
    rows: list[dict[str, Any]] | None    # Multi-row output (mutually exclusive with row)
    reason: dict[str, Any] | None        # Error details or None (success)
    retryable: bool = False              # Can this operation be retried?

    # Audit fields (set by executor, NOT by plugin)
    input_hash: str | None
    output_hash: str | None
    duration_ms: float | None

    @property
    def is_multi_row(self) -> bool:
        """True if this result contains multiple output rows."""
        return self.rows is not None

    @property
    def has_output_data(self) -> bool:
        """True if this result has any output data (single or multi-row)."""
        return self.row is not None or self.rows is not None
```

**Invariants:**
- `status == "success"` requires `has_output_data == True`
- `row` and `rows` are mutually exclusive (exactly one is set on success)
- `status == "error"` requires `reason is not None`

**Factory methods:**

```python
TransformResult.success(row)                    # Success with single output row
TransformResult.success_multi(rows)             # Success with multiple output rows
TransformResult.error(reason, retryable=False)  # Failure
```

**Multi-row usage:**

```python
# Deaggregation: 1 input → N outputs
def process(self, row, ctx) -> TransformResult:
    items = row["items"]
    return TransformResult.success_multi([
        {**row, "item": item} for item in items
    ])

# Aggregation passthrough: N inputs → N enriched outputs
def process(self, rows: list[dict], ctx) -> TransformResult:
    return TransformResult.success_multi([
        {**r, "batch_size": len(rows)} for r in rows
    ])
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
- **TransformErrorEvent** for each `TransformResult.error()`:
  - `run_id`, `token_id`, `transform_id`
  - `row` (input row data)
  - `error_details` (from TransformResult.error())
  - `destination` (sink name or "discard")
  - `input_hash` (for traceability)
  - `timestamp`

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

The following are **engine-level operations** that coordinate token flow through the DAG. They operate on **wrapped data** (tokens, routing metadata) rather than row contents directly.

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA FLOW ARCHITECTURE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PLUGINS (touch row contents)                                    │
│  ────────────────────────────                                    │
│  Source ──► [row data] ──► Transform ──► [row data] ──► Sink    │
│                                    │                              │
│                                    ▼                              │
│  ROUTING DECISIONS                                                │
│  ─────────────────                                               │
│  Gate: "where does this token go?" (config OR plugin)            │
│                                                                   │
│  STRUCTURAL OPERATIONS (config-driven only)                      │
│  ──────────────────────────────────────────                      │
│  Fork: "copy token to parallel paths"                            │
│  Coalesce: "merge tokens from paths"                             │
│  Aggregation: "batch tokens until trigger"                       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### Gate (Routing Decision)

**Purpose:** Evaluate a condition on row data and decide where the token goes next.

**Key property:** Gates make routing decisions. They may optionally modify row data (e.g., adding routing metadata).

**Two approaches are supported:**

| Approach | Use When | Implementation |
|----------|----------|----------------|
| **Config Expression** | Simple field comparisons (`score > 0.8`) | YAML `condition` string |
| **Plugin Gate** | Complex logic (ML models, multi-field analysis, external lookups) | `BaseGate` subclass |

Choose config expressions for simple, readable routing. Choose plugin gates when you need:
- Stateful evaluation (counters, caches)
- External service calls
- Complex business logic that doesn't fit in an expression
- Reusable routing logic across pipelines

#### Approach 1: Config Expression Gates

Config expressions are ideal for simple, declarative routing.

**Configuration:**

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

**How It Works:**

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

#### Approach 2: Plugin Gates

Plugin gates are system code for complex routing logic that doesn't fit in config expressions.

**Required Attributes:**

```python
name: str                          # Plugin identifier (e.g., "safety_check", "ml_router")
input_schema: type[PluginSchema]   # Schema of incoming rows
output_schema: type[PluginSchema]  # Schema of outgoing rows (usually same as input)
node_id: str | None                # Set by orchestrator after registration
determinism: Determinism           # DETERMINISTIC, NON_DETERMINISTIC, or EXTERNAL_CALL
plugin_version: str                # Semantic version for reproducibility
```

**Required Methods:**

```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize with configuration.

    Called once at pipeline construction.
    Validate config here - fail fast if misconfigured.
    """

def evaluate(
    self,
    row: dict[str, Any],
    ctx: PluginContext,
) -> GateResult:
    """Evaluate a row and decide routing.

    Args:
        row: Input row matching input_schema
        ctx: Plugin context

    Returns:
        GateResult with routing decision
    """

def close(self) -> None:
    """Release resources.

    Called after on_complete() or on error.
    """
```

**Required Lifecycle Hooks:**

```python
def on_start(self, ctx: PluginContext) -> None:
    """Called before any rows are processed."""

def on_complete(self, ctx: PluginContext) -> None:
    """Called after all rows are processed."""
```

**GateResult Contract:**

```python
@dataclass
class GateResult:
    row: dict[str, Any]        # Row data (may be modified)
    action: RoutingAction      # Where the token goes

# RoutingAction options:
RoutingAction.continue_()              # Continue to next node in pipeline
RoutingAction.route("sink_name")       # Route to named sink
RoutingAction.fork(["path1", "path2"]) # Fork to multiple parallel paths
```

**Example Plugin Gate:**

```python
class SafetyGate(BaseGate):
    """Route suspicious content to review queue.

    Uses ML model for complex content analysis that
    can't be expressed as a simple field comparison.
    """

    name = "safety_check"
    input_schema = ContentSchema
    output_schema = ContentSchema
    determinism = Determinism.EXTERNAL_CALL  # ML model is external
    plugin_version = "1.2.0"

    def __init__(self, config: dict[str, Any]) -> None:
        self._threshold = config.get("threshold", 0.7)
        self._model = None  # Lazy load

    def on_start(self, ctx: PluginContext) -> None:
        from mycompany.ml import SafetyClassifier
        self._model = SafetyClassifier.load()

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        # Complex analysis that can't be a config expression
        score = self._model.predict(row["content"])

        if score > self._threshold:
            # Add audit metadata before routing
            row["safety_score"] = score
            row["flagged_at"] = ctx.run_started_at.isoformat()
            return GateResult(row=row, action=RoutingAction.route("review_queue"))

        return GateResult(row=row, action=RoutingAction.continue_())

    def close(self) -> None:
        if self._model:
            self._model.unload()
```

**Configuration (using plugin gate):**

```yaml
pipeline:
  - source: content_feed

  - gate: safety_check           # References plugin by name
    threshold: 0.8               # Plugin-specific config
    routes:                      # Route labels must match RoutingAction.route() calls
      review_queue: review_sink

  - transform: enrich

  - sink: output
```

**When to Use Plugin Gates vs Config Expressions:**

| Scenario | Use |
|----------|-----|
| `row['score'] > threshold` | Config expression |
| Multi-field weighted scoring | Plugin gate |
| External API call for validation | Plugin gate |
| ML model inference | Plugin gate |
| Stateful routing (rate limiting, quotas) | Plugin gate |
| Simple field existence check | Config expression |
| Complex business rules | Plugin gate |

#### Gate Audit Trail

Both approaches record:
- Condition evaluated: expression text OR plugin name + version
- Route chosen: label + destination
- Timing: evaluation duration
- For plugin gates: any row modifications

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

**Purpose:** Collect multiple tokens until a trigger fires, then process as batch.

**Key property:** Converts N input tokens into M output tokens (often N→1 for aggregates).

**Architecture:** Aggregation is **fully structural** - the engine owns the buffer and decides when to flush. There is no plugin-level aggregation protocol. When you need batch processing:

1. Configure an aggregation node in the pipeline
2. Use a batch-aware transform (`is_batch_aware = True`) at that node
3. Engine buffers rows and calls `transform.process(rows: list[dict])` when trigger fires

#### Configuration

```yaml
pipeline:
  - source: events

  # Aggregation configuration
  aggregations:
    - node_id: batch_stats      # Links to transform below
      trigger:
        count: 100              # Fire after 100 rows
        timeout_seconds: 3600   # Or after 1 hour
        # condition: "row['type'] == 'flush_signal'"  # Optional: trigger on special row

  transforms:
    - plugin: summary_transform
      node_id: batch_stats      # Same as aggregation node_id
      # Transform must have is_batch_aware = True
```

#### Output Mode

Aggregation supports three output modes that determine how batch results are handled:

```yaml
aggregations:
  - node_id: batch_stats
    trigger: { count: 100 }
    output_mode: single      # default: N inputs → 1 output
```

| Mode | Input → Output | Token Handling | While Buffering | Use Case |
|------|----------------|----------------|-----------------|----------|
| `single` | N → 1 | Triggering token reused | `CONSUMED_IN_BATCH` | Aggregation (sum, count, mean) |
| `passthrough` | N → N | Same tokens preserved | `BUFFERED` | Batch enrichment |
| `transform` | N → M | New tokens created | `CONSUMED_IN_BATCH` | Group-by, splitting |

**Outcome semantics:**

- **`single`** (default): Classic aggregation. N rows become 1 aggregated row. All input tokens are terminal (`CONSUMED_IN_BATCH`). The triggering token is reused for the output.

- **`passthrough`**: Batch enrichment. N rows become N enriched rows with the same token IDs. Buffered tokens get `BUFFERED` (non-terminal) while waiting, then reappear as `COMPLETED` on flush. Transform must return `success_multi()` with exactly N rows.

- **`transform`**: Batch transformation. N rows become M rows with new tokens. All input tokens are terminal (`CONSUMED_IN_BATCH`). New tokens are created via `expand_token()` with parent linkage to the triggering token.

**Error handling:** All modes are atomic - if the transform returns `error`, ALL buffered rows fail together.

#### Trigger Types

| Trigger | Fires When |
|---------|------------|
| `count` | N tokens accumulated |
| `timeout_seconds` | Duration elapsed since batch start |
| `condition` | Row matches expression |
| `end_of_source` | Source exhausted (implicit, always checked) |

Multiple triggers can be combined (first one to fire wins).

#### How It Works (Engine-Level)

1. Tokens arrive at aggregation node
2. **Engine buffers** the row internally:
   - Assigns `batch_id`
   - Records batch membership in audit trail
   - Updates trigger evaluator
   - Token outcome depends on `output_mode`:
     - `single`/`transform`: `CONSUMED_IN_BATCH` (terminal)
     - `passthrough`: `BUFFERED` (non-terminal, will reappear as `COMPLETED`)
3. Engine checks trigger conditions
4. When trigger fires:
   - Engine retrieves buffered rows as `list[dict]`
   - Engine calls `transform.process(rows, ctx)` with the batch
   - Batch state: `draft` → `executing` → `completed`
   - Output handling depends on `output_mode`:
     - `single`: Triggering token continues with aggregated data
     - `passthrough`: All buffered tokens continue with enriched data
     - `transform`: New tokens created via `expand_token()` continue
   - Buffer is cleared for next batch

**Important:** The engine owns the buffer, not the transform. Transforms simply receive `list[dict]` and return a result. This enables:
- Crash recovery (buffers are checkpointed)
- Consistent trigger evaluation
- Clean audit trail

#### Batch Lifecycle

```
Batch B1 (draft) - Engine buffers rows
    │
    ├── Buffer T1.row_data → batch: [row1]
    ├── Buffer T2.row_data → batch: [row1, row2]
    ├── Buffer T3.row_data → batch: [row1, row2, row3]
    │
    ▼ (trigger: count >= 3)
    │
Batch B1 (executing)
    │
    ▼ Engine calls transform.process([row1, row2, row3], ctx)
    │
Batch B1 (completed)
    │
    └──► Output Token T4
         row_data: {count: 3, sum: 150, avg: 50}

Input tokens T1, T2, T3 → terminal state: CONSUMED_IN_BATCH
```

#### Crash Recovery

The engine persists buffer state in checkpoints:
- `get_checkpoint_state()` serializes buffered rows and batch metadata
- `restore_from_checkpoint()` restores buffers after crash
- Trigger evaluators resume from correct count

This means in-progress batches survive crashes and can be resumed.

#### Audit Trail

- Batch created: batch_id, trigger config, aggregation_node_id
- Batch membership: which tokens belong to which batch (ordinal position)
- Batch state transitions: draft → executing → completed/failed
- Trigger event: which condition fired, when
- Transform input/output hashes for the batch call

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
| `load` (Source) | Validation failure → quarantine via `on_validation_failure`; Code bug → crash |
| `process` (Transform) | Processing error → route via `on_error`; Code bug → crash |
| `write` (Sink) | Data error → return error result; Code bug → raise |

### Error Routing Summary

| Plugin Type | Config Field | Required? | On Missing Config + Error |
|-------------|--------------|-----------|---------------------------|
| **Source** | `on_validation_failure` | Yes | N/A (config validation fails) |
| **Transform** | `on_error` | No | `ConfigurationError` - pipeline crashes |
| **Sink** | N/A | N/A | Sinks don't route errors |

---

## Determinism Declaration

Plugins declare their determinism level:

```python
class Determinism(Enum):
    DETERMINISTIC = "deterministic"          # Same input → same output, always
    SEEDED = "seeded"                        # Same input → same output given captured seed
    IO_READ = "io_read"                      # Reads from external state (files/env/time)
    IO_WRITE = "io_write"                    # Writes side effects (files/db)
    EXTERNAL_CALL = "external_call"          # Network/API calls (record request/response)
    NON_DETERMINISTIC = "non_deterministic"  # Cannot be reproduced (record outputs)
```

**Implications:**
- `DETERMINISTIC`: Safe to replay; verify can recompute and compare
- `SEEDED`: Capture and replay with same seed
- `IO_READ`: Capture what was read (inputs/metadata) for replay/verify
- `IO_WRITE`: Side effects; replay requires care and idempotency
- `EXTERNAL_CALL`: Record request/response for replay; verify can diff responses
- `NON_DETERMINISTIC`: Record outputs; verify cannot recompute deterministically

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
| 1.6 | 2026-01-20 | Gate documentation: clarified two approaches (config expressions AND plugin gates via `BaseGate`), added `GateResult` contract, plugin gate lifecycle, when-to-use guidance |
| 1.5 | 2026-01-19 | Multi-row output: `creates_tokens` attribute, `TransformResult.success_multi()`, `rows` field, aggregation `output_mode` (single/passthrough/transform), `BUFFERED` and `EXPANDED` outcomes |
| 1.4 | 2026-01-19 | Batch-aware transforms (`is_batch_aware`), structural aggregation (engine owns buffers), crash recovery for batches |
| 1.3 | 2026-01-17 | Source `on_validation_failure` (required), Transform `on_error` (optional), QuarantineEvent, TransformErrorEvent |
| 1.2 | 2026-01-17 | Three-tier trust model, coercion rules by plugin type, type-safe ≠ operation-safe |
| 1.1 | 2026-01-17 | Add content_hash requirements, expression safety, engine concerns |
| 1.0 | 2026-01-17 | Initial contract - Source, Transform, Sink + System Operations |
