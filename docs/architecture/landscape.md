# Landscape System Architecture

> **The Audit Backbone of ELSPETH**

The Landscape system is ELSPETH's audit trail infrastructure, designed for **high-stakes accountability**. Every decision in a pipeline must be traceable to its source data, configuration, and code version. The system is built so that "I don't know what happened" is never an acceptable answer for any output.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [System Overview](#system-overview)
3. [Architecture Layers](#architecture-layers)
4. [Database Schema](#database-schema)
5. [Core Components](#core-components)
6. [Serialization Pipeline](#serialization-pipeline)
7. [Engine Integration](#engine-integration)
8. [Checkpoint and Recovery](#checkpoint-and-recovery)
9. [MCP Analysis Server](#mcp-analysis-server)
10. [Query Patterns](#query-patterns)
11. [File Reference](#file-reference)

---

## Design Philosophy

### Three-Tier Trust Model

Landscape enforces strict trust boundaries for data handling:

| Tier | Data Type | Trust Level | Error Handling |
|------|-----------|-------------|----------------|
| **Tier 1** | Audit Database | Full Trust | **Crash immediately** on any anomaly |
| **Tier 2** | Pipeline Data | Elevated Trust | Types valid, operations may fail |
| **Tier 3** | External Data | Zero Trust | Validate, coerce, quarantine |

**Tier 1 Principle**: The audit database is OUR data. If we read garbage from our own database, something catastrophic happened (bug, corruption, tampering). We crash immediately because:
- Silent coercion is evidence tampering
- An auditor asking "why did row 42 get routed here?" must get the truth
- A confident wrong answer is worse than a crash

### Core Invariants

1. **Complete Traceability**: Every output can explain its lineage
2. **Immutable Integrity**: Hashes survive payload deletion
3. **No Silent Drops**: Every row reaches exactly one terminal state
4. **Crash-on-Corruption**: Invalid audit data crashes the system
5. **Deterministic Hashing**: RFC 8785 canonicalization ensures reproducibility

---

## System Overview

### Metrics

| Metric | Value |
|--------|-------|
| **Total Files** | ~12 Python modules |
| **Lines of Code** | ~5,500 |
| **Database Tables** | 17 |
| **Repository Classes** | ~15 |
| **MCP Analysis Tools** | 25 |
| **Public API Exports** | 50+ |

> *Metrics are approximate. Run `find src/elspeth/core/landscape -name "*.py" | xargs wc -l` for current counts.*

### Key Capabilities

- **Complete lineage queries** via `explain()` function
- **Atomic fork/coalesce/expand** operations
- **Thread-safe call indexing** for concurrent transforms
- **Content-addressable payload storage** with integrity verification
- **Compliance-ready export** with optional HMAC signing
- **Crash recovery** via checkpoint/resume system
- **Real-time debugging** via MCP analysis server

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ENGINE LAYER (orchestrator.py, processor.py, executors.py)             │
│  Uses LandscapeRecorder to record all audit events                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LANDSCAPE RECORDER (2,809 lines, 60+ methods)                          │
│  High-level API for recording: runs, rows, tokens, states, calls, etc.  │
│  Thread-safe call indexing, atomic fork/coalesce/expand operations      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  REPOSITORY LAYER   │  │  CANONICAL JSON     │  │  PAYLOAD STORE      │
│  15 classes         │  │  RFC 8785 + SHA-256 │  │  Content-addressable│
│  Row→Object convert │  │  NaN/Inf rejection  │  │  Large blob storage │
│  Tier 1 validation  │  │  Type normalization │  │  Hash verification  │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATABASE OPS (_database_ops.py) - Boilerplate reduction                │
│  execute_fetchone, execute_fetchall, execute_insert, execute_update     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LANDSCAPE DB (database.py) - Connection management                     │
│  SQLite (WAL mode) / PostgreSQL, Schema validation, Context managers    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SCHEMA (schema.py) - 17 SQLAlchemy Core tables                         │
│  Composite PKs, Composite FKs, XOR constraints, Partial unique indexes  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### Schema Design Principles

- **SQLAlchemy Core** (not ORM) for explicit query control
- **Multi-backend support**: SQLite for development, PostgreSQL for production
- **Composite primary keys** for reusable node configurations across runs
- **Denormalized run_id** on node_states for efficient queries
- **XOR constraints** for mutually exclusive parent relationships
- **Partial unique indexes** for conditional uniqueness

### Table Overview

#### Core Tables

| Table | Purpose | Key Pattern |
|-------|---------|-------------|
| **runs** | Pipeline execution metadata | PK: `run_id` |
| **nodes** | Plugin instances | **Composite PK**: `(node_id, run_id)` |
| **edges** | DAG connectivity | Composite FK to nodes |

#### Data Flow Tables

| Table | Purpose | Key Relationships |
|-------|---------|-------------------|
| **rows** | Source data entries | FK to runs, Composite FK to nodes |
| **tokens** | Row instances in DAG paths | FK to rows |
| **token_parents** | Fork/join lineage | Multi-parent support |
| **token_outcomes** | Terminal states | Partial unique index (1 terminal per token) |

#### Execution Tables

| Table | Purpose | Special Constraints |
|-------|---------|---------------------|
| **node_states** | Processing records | Discriminated union (OPEN/PENDING/COMPLETED/FAILED) |
| **operations** | Source/sink I/O | XOR parent option for calls (alternative to state_id) |
| **calls** | External calls (LLM, HTTP, SQL) | **XOR constraint**: `state_id` OR `operation_id`, never both |
| **routing_events** | Gate routing decisions | FK to node_states and edges |

#### Operations Table Detail

The `operations` table tracks I/O operations performed by sources and sinks, providing call attribution for external calls made outside of transform processing:

| Operation Type | Purpose | Examples |
|----------------|---------|----------|
| `source_load` | Data ingestion during source iteration | Blob downloads, file reads, API fetches |
| `sink_write` | Output persistence during sink execution | File writes, database inserts, API posts |

**XOR Call Attribution**: External calls (`calls` table) must have exactly one parent:
- **Transform calls** → `state_id` parent (processing a row through a node)
- **Source/sink calls** → `operation_id` parent (I/O operations)

#### Aggregation Tables

| Table | Purpose |
|-------|---------|
| **batches** | Aggregation groups with trigger_type |
| **batch_members** | Tokens in batch with ordinal |
| **batch_outputs** | Batch outputs (tokens or artifacts) |

#### Error & Recovery Tables

| Table | Purpose |
|-------|---------|
| **validation_errors** | Source validation failures (quarantined rows) |
| **transform_errors** | Transform execution failures |
| **checkpoints** | Crash recovery checkpoints |
| **artifacts** | Sink outputs with content_hash |

### Critical Design Patterns

#### 1. Composite Primary Key Pattern

The `nodes` table uses composite PK `(node_id, run_id)` because the same plugin configuration can run multiple times. Node IDs are deterministic (based on config hash), so they're reused across runs.

```python
# WRONG - Ambiguous join when node_id is reused across runs
.join(nodes_table, node_states_table.c.node_id == nodes_table.c.node_id)

# CORRECT - Use denormalized run_id on node_states directly
.where(node_states_table.c.run_id == run_id)

# CORRECT - If you must join nodes, use BOTH keys
.join(
    nodes_table,
    (node_states_table.c.node_id == nodes_table.c.node_id) &
    (node_states_table.c.run_id == nodes_table.c.run_id)
)
```

#### 2. XOR Constraint (Exactly One Parent)

The `calls` table enforces that each call has exactly one parent - either a `node_state` (transform processing) OR an `operation` (source/sink I/O), never both:

```sql
CHECK (
  (state_id IS NOT NULL AND operation_id IS NULL) OR
  (state_id IS NULL AND operation_id IS NOT NULL)
)
```

#### 3. Discriminated Union (NodeState)

`NodeStateRepository.load()` validates invariants based on status:

| Status | Required Fields | Forbidden Fields |
|--------|-----------------|------------------|
| **OPEN** | input_hash, started_at | output_hash, completed_at, duration_ms |
| **PENDING** | completed_at, duration_ms | output_hash |
| **COMPLETED** | output_hash, completed_at, duration_ms | error_json |
| **FAILED** | completed_at, duration_ms | — |

Violations crash immediately (Tier 1 trust model).

#### 4. Partial Unique Index (Token Outcomes)

Exactly one terminal outcome per token, enforced via partial unique index:

```python
Index(
    "ix_token_outcomes_terminal_unique",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.is_terminal == 1),
    postgresql_where=(token_outcomes_table.c.is_terminal == 1),
)
```

---

## Core Components

### LandscapeRecorder

**File**: `src/elspeth/core/landscape/recorder.py` (~2,800 lines)

The primary interface for all audit operations, providing 60+ methods organized into functional groups:

#### Run Management
- `begin_run()` - Start new pipeline run with configuration and hashing
- `complete_run()` - Mark run as complete with final status
- `get_run()` / `list_runs()` - Retrieve run records

#### Pipeline Registration
- `register_node()` - Register plugin instances with config hash, determinism, schema
- `register_edge()` - Register connections between nodes

#### Data Flow Recording
- `create_row()` - Create source row with hash and optional payload ref
- `create_token()` - Create row instance token
- `fork_token()` - Create fork group for fan-out (ATOMIC)
- `coalesce_tokens()` - Create join for convergence (ATOMIC)
- `expand_token()` - Deaggregation expansion (ATOMIC)

#### Processing States
- `begin_node_state()` - Start processing a row at a node
- `complete_node_state()` - Complete state with output

#### External Calls
- `allocate_call_index()` - Thread-safe call numbering per state_id
- `record_call()` - Record HTTP/LLM/database call

#### Thread Safety

```python
def allocate_call_index(self, state_id: str) -> int:
    """Thread-safe call index allocation."""
    with self._call_index_lock:
        next_idx = self._call_indices.get(state_id, 0)
        self._call_indices[state_id] = next_idx + 1
        return next_idx
```

Ensures `UNIQUE(state_id, call_index)` across all concurrent calls.

### Repository Layer

**File**: `src/elspeth/core/landscape/repositories.py` (577 lines)

15 repository classes that convert SQLAlchemy rows to domain objects with **strict Tier 1 validation**:

| Repository | Purpose | Validation |
|------------|---------|------------|
| RunRepository | Run records | Crashes on invalid RunStatus |
| NodeRepository | Plugin instances | Crashes on invalid NodeType, Determinism |
| TokenOutcomeRepository | Terminal states | Validates is_terminal is 0 or 1 |
| NodeStateRepository | Processing records | Validates discriminated union invariants |

**Example: Tier 1 Crash Semantics**

```python
class RunRepository:
    def load(self, row: Any) -> Run:
        # No try/except - crashes on invalid enum value
        return Run(
            run_id=row.run_id,
            status=RunStatus(row.status),  # Crashes if invalid
        )
```

### LandscapeDB

**File**: `src/elspeth/core/landscape/database.py` (252 lines)

Connection management with:
- **SQLite WAL mode** for better concurrency
- **Foreign keys enforcement** at database level
- **Schema validation** on startup
- **Context manager** for automatic commit/rollback

```python
@contextmanager
def connection(self) -> Iterator[Connection]:
    with self.engine.begin() as conn:  # Auto-commit/rollback
        yield conn
```

### Lineage Query System

**File**: `src/elspeth/core/landscape/lineage.py` (204 lines)

Complete lineage query via `explain()` function:

```python
@dataclass
class LineageResult:
    token: Token                    # The token being explained
    source_row: RowLineage          # Original source with resolved payload
    node_states: list[NodeState]    # All processing states in order
    routing_events: list[RoutingEvent]  # All routing decisions
    calls: list[Call]               # All external calls made
    parent_tokens: list[Token]      # Parents (for forks/joins)
    validation_errors: list[ValidationErrorRecord]
    transform_errors: list[TransformErrorRecord]
    outcome: TokenOutcome | None    # Terminal outcome
```

---

## Serialization Pipeline

### Two-Phase Canonicalization

ELSPETH uses RFC 8785/JCS (JSON Canonicalization Scheme) for deterministic hashing:

```
Input Data
    │
    ▼
Phase 1: NORMALIZATION (_normalize_for_canonical)
    ├─ NumPy types → Python primitives
    ├─ Pandas types → primitives
    ├─ datetime → ISO 8601 UTC string
    ├─ Decimal → string (preserves precision)
    ├─ bytes → {"__bytes__": base64_string}
    └─ [REJECT: NaN, Infinity, non-finite Decimal]
    │
    ▼
Phase 2: RFC 8785 SERIALIZATION (rfc8785.dumps)
    ├─ Sorted keys (lexicographic)
    ├─ No whitespace (compact)
    └─ Deterministic across Python versions
    │
    ▼
Output: Canonical JSON string → SHA-256 hash
```

### Type Conversion Table

| Input Type | Output | Notes |
|-----------|--------|-------|
| `float` / `np.floating` | `float` | NaN/Infinity **REJECTED** |
| `np.integer` | `int` | Type-safe conversion |
| `np.ndarray` | `list` | Element-wise conversion; NaN/Infinity **REJECTED** |
| `pd.Timestamp` | ISO 8601 string | Converted to UTC |
| `pd.NaT` / `pd.NA` | `None` | Intentional missing values |
| `datetime` (naive) | ISO 8601 UTC string | Assumed UTC |
| `datetime` (aware) | ISO 8601 UTC string | Converted to UTC |
| `bytes` | `{"__bytes__": base64}` | Base64-wrapped |
| `Decimal` | `str` | Rejects non-finite |

### NaN/Infinity Rejection

Strict rejection per audit integrity requirements:

```python
# Rejected at first encounter
if math.isnan(value) or math.isinf(value):
    raise ValueError("Cannot canonicalize non-finite float")
```

This is defense-in-depth for audit integrity - NaN/Infinity in audit data would make hashes non-deterministic.

### Hash Storage

| Table | Hash Field | Purpose |
|-------|-----------|---------|
| runs | config_hash | DAG topology immutability |
| rows | source_data_hash | Payload integrity verification |
| node_states | input_hash, output_hash | Transform audit |
| calls | request_hash, response_hash | External call audit |
| artifacts | content_hash | Sink output audit |
| checkpoints | upstream_topology_hash | Resume validation |

---

## Engine Integration

The engine integrates with Landscape at **9 distinct layers**:

| Layer | Component | LandscapeRecorder Methods |
|-------|-----------|---------------------------|
| 1 | Run Management | `begin_run()`, `complete_run()`, `register_node()`, `register_edge()` |
| 2 | Row & Token Creation | `create_row()`, `create_token()` |
| 3 | Node State Recording | `begin_node_state()`, `complete_node_state()` |
| 4 | Routing Events | `record_routing_event()` |
| 5 | Terminal Outcomes | `record_token_outcome()` |
| 6 | Fork/Join Lineage | `fork_token()`, `coalesce_tokens()`, `expand_token()` |
| 7 | Batch/Aggregation | `create_batch()`, `add_batch_member()`, `complete_batch()` |
| 8 | External Calls | `record_call()` (XOR: state_id or operation_id) |
| 9 | Error Recording | `record_validation_error()`, `record_transform_error()` |

### Atomic Operations

Fork, coalesce, and expand operations are atomic - all changes in one transaction:

```python
def fork_token(self, parent_token_id: str, branches: list[str]) -> tuple[list[Token], str]:
    """Fork parent token into N children.

    ATOMIC: Creates children, parent relationships, and FORKED outcome
    in one transaction. If any step fails, all fail.
    """
    with self._db.connection() as conn:
        # 1. Create child tokens
        for branch_name in branches:
            conn.execute(tokens_table.insert().values(...))
            conn.execute(token_parents_table.insert().values(...))

        # 2. Record parent FORKED outcome in SAME transaction
        conn.execute(token_outcomes_table.insert().values(
            outcome=RowOutcome.FORKED.value,
            is_terminal=1,
            expected_branches_json=json.dumps(branches),
        ))
    # Transaction auto-commits here
```

> *Simplified for illustration. See `recorder.py:fork_token()` for complete implementation.*

### Context State Management

Executors manage context state for call attribution:

```python
# Transform executor: set state_id before transform.process()
ctx._state_id = state_id
ctx._operation_id = None

# Sink executor: clear state_id, set operation_id for I/O calls
ctx._state_id = None
ctx._operation_id = operation_id
```

This ensures calls are attributed to the correct parent.

---

## Checkpoint and Recovery

### Checkpoint Creation

Checkpoints are created after sink writes to ensure durability:

```
Token reaches sink → Sink writes output → _maybe_checkpoint()
    ├─ Compute full topology hash (ALL nodes + edges)
    ├─ Capture aggregation state (if applicable)
    └─ Store in checkpoints_table with sequence_number
```

> **Note**: Sink writes also create `operations` records (type: `sink_write`). Both the checkpoint and the operation record are created in the same transactional flow, ensuring consistency between crash recovery state and the audit trail.

**Checkpoint frequency options:**
- `frequency=1`: Every row (default)
- `frequency=N`: Every N rows
- `frequency=0`: Only at aggregation boundaries

### Resume Flow

```
1. VALIDATE COMPATIBILITY
   ├─ Checkpoint node exists in current graph?
   ├─ Node config unchanged (hash match)?
   └─ Full topology unchanged? (prevents config drift)

2. GET UNPROCESSED ROWS (via token_outcomes)
   └─ Find rows with non-terminal, non-delegation tokens
       (Excludes: FORKED, EXPANDED which delegate to children)

3. RETRIEVE ROW DATA WITH TYPE RESTORATION
   ├─ Load from payload_store
   ├─ Verify hash integrity (crash if mismatch)
   └─ Re-validate through source_schema_class
       (JSON string → datetime/Decimal)

4. PROCESS UNPROCESSED ROWS
   ├─ Restore aggregation state
   ├─ Process through DAG
   ├─ Create checkpoints as rows complete
   └─ Finalize run
```

### Type Fidelity Preservation

JSON serialization loses type information. Type restoration on resume:

```
Original: {"created_at": datetime(2024, 1, 1)}
    ↓ canonical_json()
Stored: {"created_at": "2024-01-01T00:00:00+00:00"}
    ↓ json.loads()
Loaded: {"created_at": "2024-01-01T00:00:00+00:00"}  # String!
    ↓ source_schema_class.model_validate()
Restored: {"created_at": datetime(2024, 1, 1)}  # Type restored!
```

The source schema is stored in `runs.source_schema_json` for this purpose.

### Topology Validation

Full topology hash includes ALL nodes + edges:
- Prevents configuration drift across resume
- Any change (upstream, downstream, sibling) invalidates checkpoint
- Enforces: one run_id = one pipeline configuration

---

## MCP Analysis Server

**File**: `src/elspeth/mcp/server.py` (~2,100 lines)

A read-only analysis server providing 25 tools for debugging and investigation.

### Emergency Diagnostic Tools (Use First)

| Tool | Purpose | Use When |
|------|---------|----------|
| `diagnose()` | Scan for failed/stuck runs, high error rates | Something is obviously broken |
| `get_failure_context(run_id)` | Deep dive on specific failure | You have a failed run_id |
| `get_recent_activity(minutes)` | Timeline of recent activity | Need to understand what happened |

### Core Query Tools

| Tool | Purpose |
|------|---------|
| `list_runs(limit, status)` | List runs with optional status filter |
| `get_run_summary(run_id)` | Comprehensive run statistics |
| `explain_token(run_id, token_id)` | Complete lineage for a row |
| `get_errors(run_id, error_type)` | Validation and transform errors |
| `query(sql)` | Ad-hoc read-only SQL (SELECT only) |

### Analysis Tools

| Tool | Purpose |
|------|---------|
| `get_dag_structure(run_id)` | Mermaid diagram of pipeline |
| `get_performance_report(run_id)` | Bottleneck identification |
| `get_error_analysis(run_id)` | Error pattern grouping |
| `get_llm_usage_report(run_id)` | LLM call statistics |
| `get_outcome_analysis(run_id)` | Terminal state distribution |

### Investigation Workflows

**Scenario 1: "Something Failed"**
```
diagnose() → Identifies failed run_id
get_failure_context(run_id) → Shows failed states and errors
explain_token(run_id, token_id) → Complete lineage of specific row
```

**Scenario 2: "Pipeline is Slow"**
```
get_run_summary(run_id) → Overall statistics
get_performance_report(run_id) → Bottleneck nodes
get_llm_usage_report(run_id) → If LLM transforms suspected
```

**Scenario 3: "Data Quality Issues"**
```
get_run_summary(run_id) → Error counts
get_error_analysis(run_id) → Group by source/type
get_errors(run_id, "validation") → Sample failing rows
```

### Database Discovery

The MCP server auto-discovers databases:

1. Search current directory (up to 5 levels)
2. Prioritize by name: `audit.db` in `runs/` → highest
3. Sort by modification time
4. Interactive selection if multiple found

---

## Query Patterns

### Safe Single-Row Queries

```python
query = select(runs_table).where(runs_table.c.run_id == run_id)
row = ops.execute_fetchone(query)  # Crashes if multiple rows
run = RunRepository().load(row) if row else None
```

### Multi-Row with Repository Conversion

```python
query = select(batches_table).where(batches_table.c.run_id == run_id)
rows = ops.execute_fetchall(query)
batches = [BatchRepository().load(row) for row in rows]
```

### Composite Key Queries

```python
# CORRECT: Use denormalized run_id directly
query = (
    select(node_states_table)
    .where(node_states_table.c.run_id == run_id)
    .where(node_states_table.c.token_id == token_id)
)

# CORRECT: Composite join if you must access nodes columns
query = (
    select(node_states_table, nodes_table.c.plugin_name)
    .join(
        nodes_table,
        (node_states_table.c.node_id == nodes_table.c.node_id) &
        (node_states_table.c.run_id == nodes_table.c.run_id)
    )
    .where(node_states_table.c.run_id == run_id)
)
```

### Index Coverage

| Query Pattern | Index Used |
|---------------|------------|
| `WHERE run_id = ?` | `ix_nodes_run_id`, `ix_edges_run_id`, etc. |
| `WHERE run_id = ? AND status = ?` | `ix_batches_run_status` (composite) |
| `WHERE token_id = ?` | `ix_node_states_token` |
| `WHERE state_id = ?` | `ix_calls_state` |
| `WHERE routing_group_id = ?` | `ix_routing_events_group` |

---

## File Reference

### Core Landscape Module

| File | Purpose |
|------|---------|
| `recorder.py` | Main recording API (60+ methods) |
| `schema.py` | 17 SQLAlchemy Core table definitions |
| `repositories.py` | ~15 repository classes for row→object conversion |
| `database.py` | Connection management (SQLite WAL, PostgreSQL) |
| `exporter.py` | Compliance export with optional HMAC signing |
| `formatters.py` | Export formatters (CSV, JSON, Lineage text) |
| `lineage.py` | Complete lineage query (`explain()`) |
| `reproducibility.py` | Reproducibility grade computation |
| `row_data.py` | State discrimination for row retrieval |
| `_database_ops.py` | Boilerplate reduction helper |
| `_helpers.py` | ID generation, timestamps |
| `__init__.py` | Public API exports |

> *Line counts omitted to prevent staleness. Use `wc -l` on individual files for current sizes.*

### Checkpoint Subsystem

| File | Purpose |
|------|---------|
| `core/checkpoint/manager.py` | Checkpoint CRUD operations |
| `core/checkpoint/recovery.py` | Resume point determination |
| `core/checkpoint/compatibility.py` | Topology validation |

### MCP Server

| File | Purpose |
|------|---------|
| `mcp/server.py` | 25 analysis tools for debugging and investigation |
| `mcp/__init__.py` | Public API exports |

### Related Modules

| File | Purpose |
|------|---------|
| `core/canonical.py` | RFC 8785 canonicalization |
| `core/payload_store.py` | Content-addressable blob storage |
| `contracts/audit.py` | Domain model definitions |
| `contracts/checkpoint.py` | Resume contracts |

---

## Appendix: Terminal State Reference

Every token reaches exactly one terminal state:

| Outcome | Meaning | Parent Outcome |
|---------|---------|----------------|
| `COMPLETED` | Reached output sink | — |
| `ROUTED` | Sent to named sink by gate | — |
| `FORKED` | Split to multiple paths | Children continue |
| `EXPANDED` | Deaggregated to multiple tokens | Children continue |
| `CONSUMED_IN_BATCH` | Aggregated into batch | Batch continues |
| `COALESCED` | Merged in join | Merged token continues |
| `QUARANTINED` | Failed, stored for investigation | — |
| `FAILED` | Failed, not recoverable | — |

---

## Appendix: Reproducibility Grades

| Grade | Meaning | Replay Capability |
|-------|---------|-------------------|
| `FULL_REPRODUCIBLE` | All nodes deterministic or seeded | Can re-execute with same seed |
| `REPLAY_REPRODUCIBLE` | Has external calls or nondeterministic nodes | Can replay using recorded responses |
| `ATTRIBUTABLE_ONLY` | Payloads purged | Can verify via hashes, cannot replay |

After payload purge, grade degrades: `REPLAY_REPRODUCIBLE` → `ATTRIBUTABLE_ONLY`

---

*Last updated: 2026-01-31*
