# Source/Sink Audit Trail Design

**Date:** 2026-01-31
**Status:** REVISED - Schema Change Approved
**Problem:** Sources and sinks with external I/O don't record calls to Landscape or emit telemetry

---

## Executive Summary

Sources and sinks can't record external calls because `calls.state_id` requires a `node_states` entry, which requires a `token_id`, but sources CREATE tokens - they don't process them.

**Solution:** Add explicit `operations` table to model source/sink I/O as first-class audit entities. This is the architecturally correct fix - no semantic overloading, no conventions, no query fragility.

---

## Current State

```
Data Flow:         Source.load() → rows → Transforms → Sink.write()
Token Creation:    ─────────────→ HERE
Node States:                        HERE ────────────────→
Call Recording:                     HERE ────────────────→
```

Sources and sinks are OUTSIDE the token-based audit model.

---

## The Right Fix: Operations Table

### New Schema

```sql
-- New table for source/sink operations
CREATE TABLE operations (
    operation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    operation_type TEXT NOT NULL CHECK(operation_type IN ('source_load', 'sink_write')),
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    status TEXT NOT NULL CHECK(status IN ('open', 'completed', 'failed')),
    input_data_ref TEXT,      -- Payload store reference for operation input
    output_data_ref TEXT,     -- Payload store reference for operation output
    error_message TEXT,       -- Error details if failed
    duration_ms REAL,
    -- Composite FK to nodes
    FOREIGN KEY (node_id, run_id) REFERENCES nodes(node_id, run_id)
);

-- Index for efficient querying
CREATE INDEX ix_operations_run_id ON operations(run_id);
CREATE INDEX ix_operations_node_id_run_id ON operations(node_id, run_id);

-- Calls table gets operation_id column
ALTER TABLE calls ADD COLUMN operation_id TEXT REFERENCES operations(operation_id);

-- Constraint: calls must have exactly one parent (state OR operation)
ALTER TABLE calls ADD CONSTRAINT calls_has_parent
    CHECK ((state_id IS NOT NULL AND operation_id IS NULL) OR
           (state_id IS NULL AND operation_id IS NOT NULL));

-- Index for operation call lookups
CREATE INDEX ix_calls_operation_id ON calls(operation_id) WHERE operation_id IS NOT NULL;
```

**SQLite Note:** SQLite supports CHECK constraints natively when creating tables. The `ALTER TABLE ... ADD CONSTRAINT` syntax shown above is PostgreSQL-style; for SQLite, define the constraint in the initial `CREATE TABLE` statement. Since we have no existing data to migrate, this is just a matter of correct DDL in `schema.py`.

**Note on call_index uniqueness:** The existing `(state_id, call_index)` uniqueness is enforced by `call_id` generation pattern. For operations, uniqueness is similarly enforced via `call_id = f"call_{operation_id}_{index}"`. No additional constraint needed since `call_id` is the PK.

### Why This is Better

| Aspect | Operational Tokens (Rejected) | Operations Table (Approved) |
|--------|------------------------------|----------------------------|
| Semantic clarity | `rows` table overloaded | Clean separation |
| Query safety | Every query needs `row_index >= 0` | Existing queries unchanged |
| FK integrity | `sink_node_id` in JSON | Proper FK to nodes |
| Indexability | Can't index JSON fields | Full index support |
| Discoverability | Sign convention is implicit | Table existence is explicit |
| Future-proof | More conventions pile up | Explicit model extends cleanly |

---

## Implementation

### Schema Definition (schema.py)

```python
# === Operations (Source/Sink I/O) ===

operations_table = Table(
    "operations",
    metadata,
    Column("operation_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False, index=True),
    Column("node_id", String(64), nullable=False),
    Column("operation_type", String(32), nullable=False),  # 'source_load' | 'sink_write'
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("status", String(16), nullable=False),  # 'open' | 'completed' | 'failed'
    Column("input_data_ref", String(256)),   # Payload store reference
    Column("output_data_ref", String(256)),  # Payload store reference
    Column("error_message", Text),
    Column("duration_ms", Float),
    # Composite FK to nodes
    ForeignKeyConstraint(["node_id", "run_id"], ["nodes.node_id", "nodes.run_id"]),
)

# Update calls table - add operation_id column
# Note: SQLAlchemy Core table definition updated to include:
#   Column("operation_id", String(64), ForeignKey("operations.operation_id")),
#   CheckConstraint(
#       "(state_id IS NOT NULL AND operation_id IS NULL) OR "
#       "(state_id IS NULL AND operation_id IS NOT NULL)",
#       name="calls_has_parent"
#   ),
```

### Recorder API

```python
# In recorder.py

def begin_operation(
    self,
    run_id: str,
    node_id: str,
    operation_type: Literal["source_load", "sink_write"],
    *,
    input_data: dict[str, Any] | None = None,
) -> Operation:
    """Begin an operation for source/sink I/O.

    Operations are the source/sink equivalent of node_states - they provide
    a parent context for external calls made during load() or write().

    Args:
        run_id: Run this operation belongs to
        node_id: Source or sink node performing the operation
        operation_type: Type of operation
        input_data: Optional input context (stored via payload store)

    Returns:
        Operation with operation_id for call attribution
    """
    # Use pure UUID for operation_id - run_id + node_id can exceed 64 chars
    # (run_id=36 + node_id=45 + prefixes would be 94+ chars)
    operation_id = f"op_{uuid4().hex}"  # "op_" + 32 hex = 35 chars, well under 64

    input_ref = None
    if input_data:
        input_ref = self._payload_store.store(input_data) if self._payload_store else None

    operation = Operation(
        operation_id=operation_id,
        run_id=run_id,
        node_id=node_id,
        operation_type=operation_type,
        started_at=datetime.now(UTC),
        status="open",
        input_data_ref=input_ref,
    )

    self._db.insert(operations_table, operation.to_dict())
    return operation


def complete_operation(
    self,
    operation_id: str,
    status: Literal["completed", "failed"],
    *,
    output_data: dict[str, Any] | None = None,
    error: str | None = None,
    duration_ms: float | None = None,
) -> None:
    """Complete an operation.

    Args:
        operation_id: Operation to complete
        status: Final status
        output_data: Optional output context
        error: Error message if failed
        duration_ms: Operation duration

    Raises:
        FrameworkBugError: If operation doesn't exist or is already completed
    """
    # Validate operation exists and is open (prevent double-complete)
    current = self._db.fetch_one(
        select(operations_table.c.status)
        .where(operations_table.c.operation_id == operation_id)
    )

    if current is None:
        raise FrameworkBugError(f"Completing non-existent operation: {operation_id}")

    if current["status"] != "open":
        raise FrameworkBugError(
            f"Completing already-completed operation {operation_id}: "
            f"current status={current['status']}, new status={status}"
        )

    output_ref = None
    if output_data:
        output_ref = self._payload_store.store(output_data) if self._payload_store else None

    self._db.update(
        operations_table,
        where=operations_table.c.operation_id == operation_id,
        values={
            "completed_at": datetime.now(UTC),
            "status": status,
            "output_data_ref": output_ref,
            "error_message": error,
            "duration_ms": duration_ms,
        },
    )


def record_operation_call(
    self,
    operation_id: str,
    call_type: CallType,
    *,
    status: CallStatus,
    request_data: dict[str, Any] | None = None,
    response_data: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    provider: str | None = None,
) -> Call:
    """Record an external call made during an operation.

    This is the operation equivalent of record_call() - attributes calls
    to operations instead of node_states.
    """
    call_id = f"call_{operation_id}_{self._next_call_index(operation_id)}"

    call = Call(
        call_id=call_id,
        state_id=None,  # NOT a node_state call
        operation_id=operation_id,  # Operation call
        call_type=call_type,
        status=status,
        # ... rest of fields
    )

    self._db.insert(calls_table, call.to_dict())
    return call
```

### Operation Lifecycle Context Manager (Recommended)

To eliminate boilerplate and ensure consistent exception handling, use a context manager:

```python
# In core/operations.py

from contextlib import contextmanager
from typing import Iterator, Literal

@dataclass
class OperationHandle:
    """Mutable handle for capturing operation output within context manager."""
    operation: Operation
    output_data: dict[str, Any] | None = None


@contextmanager
def track_operation(
    recorder: LandscapeRecorder,
    run_id: str,
    node_id: str,
    operation_type: Literal["source_load", "sink_write"],
    ctx: PluginContext,
    *,
    input_data: dict[str, Any] | None = None,
) -> Iterator[OperationHandle]:
    """Context manager for operation lifecycle tracking.

    Handles:
    - Operation creation
    - Context wiring (ctx.operation_id)
    - Duration calculation
    - Exception capture with proper status
    - Guaranteed completion (even on DB failure)
    - Context cleanup (clears operation_id after completion)

    Usage:
        with track_operation(...) as handle:
            result = sink.write(rows, ctx)
            handle.output_data = {"artifact_path": result.path}  # Explicit!
    """
    operation = recorder.begin_operation(
        run_id=run_id,
        node_id=node_id,
        operation_type=operation_type,
        input_data=input_data,
    )

    handle = OperationHandle(operation=operation)
    ctx.operation_id = operation.operation_id
    start_time = time.perf_counter()
    status: Literal["completed", "failed"] = "completed"
    error_msg: str | None = None

    try:
        yield handle
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        try:
            recorder.complete_operation(
                operation_id=operation.operation_id,
                status=status,
                output_data=handle.output_data,  # Explicit field, not getattr
                error=error_msg,
                duration_ms=duration_ms,
            )
        except Exception as db_error:
            # Log critically but don't mask original exception
            logger.critical(
                "Failed to complete operation - audit trail incomplete",
                operation_id=operation.operation_id,
                db_error=str(db_error),
            )
        finally:
            # Clear operation_id to prevent accidental reuse
            ctx.operation_id = None


# Usage in orchestrator.py (source - no output data):
with track_operation(
    recorder=recorder,
    run_id=run_id,
    node_id=source_id,
    operation_type="source_load",
    ctx=ctx,
    input_data={"source_plugin": config.source.name},
) as handle:
    source_iterator = config.source.load(ctx)
    for row_index, source_item in enumerate(source_iterator):
        # ... process rows ...
    # No finally needed - context manager handles everything
    # No output_data for sources (row count tracked elsewhere)

# Usage in executors.py (sink - with output data):
with track_operation(
    recorder=recorder,
    run_id=ctx.run_id,
    node_id=sink_node_id,
    operation_type="sink_write",
    ctx=ctx,
    input_data={"sink_plugin": sink.name, "row_count": len(tokens)},
) as handle:
    artifact_info = sink.write(rows, ctx)
    handle.output_data = {"artifact_path": artifact_info.path}  # Explicit assignment!
```

### Orchestrator Integration (Source)

```python
# In orchestrator.py, BEFORE source.load()

# Begin operation for source loading
source_operation = recorder.begin_operation(
    run_id=run_id,
    node_id=source_id,
    operation_type="source_load",
    input_data={
        "source_plugin": config.source.name,
        "source_config_hash": stable_hash(config.source._raw_config),
    },
)

# Set operation_id on context for call recording
ctx.operation_id = source_operation.operation_id

source_start = time.perf_counter()
try:
    source_iterator = config.source.load(ctx)
    # ... process rows ...
    status = "completed"
except Exception as e:
    status = "failed"
    error_msg = str(e)
    raise
finally:
    duration_ms = (time.perf_counter() - source_start) * 1000
    recorder.complete_operation(
        operation_id=source_operation.operation_id,
        status=status,
        error=error_msg if status == "failed" else None,
        duration_ms=duration_ms,
    )
```

### Executor Integration (Sink)

```python
# In executors.py execute_sink()

# Begin operation for sink write
sink_operation = self._recorder.begin_operation(
    run_id=ctx.run_id,
    node_id=sink_node_id,
    operation_type="sink_write",
    input_data={
        "sink_plugin": sink.name,
        "row_count": len(tokens),
        # Don't store full token_ids list - can be massive for large sinks
        # Token lineage is queryable via: tokens WHERE run_id = ? joined to outcomes
    },
)

# Set operation_id on context
ctx.operation_id = sink_operation.operation_id

sink_start = time.perf_counter()
try:
    artifact_info = sink.write(rows, ctx)
    status = "completed"
except Exception as e:
    status = "failed"
    error_msg = str(e)
    raise
finally:
    duration_ms = (time.perf_counter() - sink_start) * 1000
    self._recorder.complete_operation(
        operation_id=sink_operation.operation_id,
        status=status,
        output_data={"artifact_path": artifact_info.path} if artifact_info else None,
        error=error_msg if status == "failed" else None,
        duration_ms=duration_ms,
    )
```

### PluginContext Update

```python
# In context.py

@dataclass
class PluginContext:
    run_id: str
    node_id: str | None = None
    state_id: str | None = None      # For transform calls
    operation_id: str | None = None  # For source/sink calls

    def record_call(
        self,
        call_type: CallType,
        status: CallStatus,
        **kwargs,
    ) -> Call | None:
        """Record an external call to the appropriate parent.

        Enforces XOR: exactly one of state_id or operation_id must be set.
        """
        # Enforce XOR at runtime - catch framework bugs early
        if self.state_id is not None and self.operation_id is not None:
            raise FrameworkBugError(
                f"record_call() called with BOTH state_id and operation_id set. "
                f"state_id={self.state_id}, operation_id={self.operation_id}. "
                f"This is a framework bug - context should have exactly one parent."
            )

        if self.state_id is not None:
            return self._recorder.record_call(
                state_id=self.state_id,
                call_type=call_type,
                status=status,
                **kwargs,
            )
        elif self.operation_id is not None:
            return self._recorder.record_operation_call(
                operation_id=self.operation_id,
                call_type=call_type,
                status=status,
                **kwargs,
            )
        else:
            # No parent context - this is a bug
            raise FrameworkBugError(
                f"record_call() called without state_id or operation_id. "
                f"Context state: run_id={self.run_id}, node_id={self.node_id}. "
                f"This is a framework bug - context should have been set by orchestrator/executor."
            )
```

### Plugin Recording (Source Example)

```python
# In AzureBlobSource.load()

def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
    start = time.perf_counter()
    try:
        blob_data = self._blob_client.download_blob().readall()
        latency_ms = (time.perf_counter() - start) * 1000

        # Record call - context.operation_id is set by orchestrator
        ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"operation": "download_blob", "path": self._blob_path},
            response_data={"size_bytes": len(blob_data)},
            latency_ms=latency_ms,
            provider="azure",
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.ERROR,
            request_data={"operation": "download_blob", "path": self._blob_path},
            error={"type": type(e).__name__, "message": str(e)},
            latency_ms=latency_ms,
            provider="azure",
        )
        raise

    # Parse and yield rows...
    for row_index, row in enumerate(self._parse_blob(blob_data)):
        yield SourceRow.valid(row)
```

---

## Query Patterns

### Get all external calls for a run (UNION approach)

```sql
-- Calls from transforms (via node_states)
SELECT c.*, 'transform' as call_source, ns.node_id
FROM calls c
JOIN node_states ns ON c.state_id = ns.state_id
WHERE ns.run_id = :run_id

UNION ALL

-- Calls from sources/sinks (via operations)
SELECT c.*, 'operation' as call_source, op.node_id
FROM calls c
JOIN operations op ON c.operation_id = op.operation_id
WHERE op.run_id = :run_id
```

### Get source I/O for a run

```sql
SELECT op.*, c.*
FROM operations op
LEFT JOIN calls c ON c.operation_id = op.operation_id
WHERE op.run_id = :run_id
  AND op.operation_type = 'source_load'
```

### Get sink I/O for a run

```sql
SELECT op.*, c.*
FROM operations op
LEFT JOIN calls c ON c.operation_id = op.operation_id
WHERE op.run_id = :run_id
  AND op.operation_type = 'sink_write'
```

### Explain token (existing query unchanged!)

```python
# explain_token(token_id) still walks:
#   token → node_states → calls
# This works UNCHANGED for data tokens
#
# For operations, add new method:
# explain_operation(operation_id) walks:
#   operation → calls
```

---

## MCP Server Updates

### New Tools

```python
# In landscape-mcp

def get_operations(run_id: str) -> list[Operation]:
    """Get all source/sink operations for a run."""
    ...

def explain_operation(run_id: str, operation_id: str) -> OperationLineage:
    """Get complete context for an operation: node, calls, timing."""
    ...
```

### Updated diagnose()

```python
def diagnose():
    """Emergency diagnostic - now includes stuck operations."""
    # ... existing checks ...

    # Check for stuck operations (open for > 1 hour)
    stuck_ops = query("""
        SELECT * FROM operations
        WHERE status = 'open'
        AND started_at < datetime('now', '-1 hour')
    """)
    if stuck_ops:
        report["stuck_operations"] = stuck_ops
```

---

## Contract Definition

```python
# In contracts/exceptions.py (or existing exceptions module)

class FrameworkBugError(Exception):
    """Raised when framework encounters internal inconsistency.

    This indicates a bug in ELSPETH itself, not user error or external failure.
    Examples: double-completing an operation, missing required context.
    """
    pass
```

```python
# In contracts/audit.py

@dataclass(frozen=True, slots=True)
class Operation:
    """Represents a source/sink I/O operation in the audit trail.

    Operations are the equivalent of node_states for sources and sinks.
    They provide a parent context for external calls made during
    source.load() or sink.write().
    """
    operation_id: str
    run_id: str
    node_id: str
    operation_type: Literal["source_load", "sink_write"]
    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["open", "completed", "failed"] = "open"
    input_data_ref: str | None = None
    output_data_ref: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None
```

---

## Implementation Checklist

### Phase 1: Schema ✅

- [x] Add `operations` table to `schema.py`
- [x] Add `operation_id` column to `calls` table
- [x] Add CHECK constraint for calls parent (XOR)
- [x] Add index `ix_calls_operation_id` for operation call lookups
- [x] Add `Operation` dataclass to `contracts/audit.py`
- [x] Add `FrameworkBugError` exception to `contracts/errors.py`

### Phase 2: Recorder ✅

- [x] Add `begin_operation()` method
- [x] Add `complete_operation()` method with status transition validation
- [x] Add `record_operation_call()` method
- [x] Add call index tracking for operations

### Phase 3: Operations Helper ✅

- [x] Add `track_operation()` context manager to `core/operations.py`
- [x] Ensure exception safety (DB failure doesn't mask original exception)

### Phase 4: Context ✅

- [x] Add `operation_id` field to `PluginContext`
- [x] Update `record_call()` to handle operation context
- [x] Raise `FrameworkBugError` if neither state_id nor operation_id set

### Phase 5: Orchestrator ✅

- [x] Create source operation before `source.load()` (use `track_operation` context manager)
- [x] Set `ctx.operation_id`
- [x] Complete operation in finally block (handled by context manager)

### Phase 6: Executors ✅

- [x] Create sink operation before `sink.write()` (use `track_operation` context manager)
- [x] Set `ctx.operation_id`
- [x] Complete operation in finally block (handled by context manager)
- [x] Remove old `states[0][1].state_id` workaround

### Phase 7: Plugins ✅

- [x] AzureBlobSource: Add `ctx.record_call()` for blob download
- [x] AzureBlobSink: Add `ctx.record_call()` for blob upload
- [x] DatabaseSink: Add `ctx.record_call()` for SQL INSERT

### Phase 8: MCP ✅

- [x] Add `list_operations()` tool (renamed from get_operations for consistency)
- [x] Add `get_operation_calls()` tool (renamed from explain_operation for clarity)
- [x] Update `diagnose()` for stuck operations
- [x] Update `get_run_summary()` to include operation counts

### Phase 9: Testing ✅

- [x] Unit: Operation lifecycle (begin → complete)
- [x] Unit: Operation call recording
- [x] Unit: PluginContext routing (state vs operation)
- [x] Unit: XOR constraint rejects calls with both/neither parent (DB level)
- [x] Unit: PluginContext.record_call() rejects both state_id AND operation_id set (runtime)
- [x] Unit: `complete_operation()` DB failure doesn't mask original exception
- [x] Unit: Status transition validation (double-complete raises FrameworkBugError)
- [x] Unit: `track_operation` context manager lifecycle (create, complete, exception handling)
- [x] Unit: BatchPendingError → pending status
- [x] Unit: Duration recording
- [x] Unit: Output data recording
- [ ] Integration: Source with HTTP calls (deferred - requires Azure source implementation)
- [ ] Integration: Sink with database calls (deferred - requires instrumented sink implementation)

**Test file:** `tests/core/landscape/test_operations.py` (26 tests)
- [ ] Integration: explain_operation() returns correct lineage
- [ ] Integration: Streaming source with long-lived operation
- [ ] Integration: Source failure before yield marks operation FAILED
- [ ] Integration: Sink partial write tracking
- [ ] Integration: Empty source (0 rows) completes operation normally

### Phase 10: Query Migration Audit ✅

Audited and updated queries that assumed `calls.state_id` is always set:

- [x] **purge.py**: Updated `find_expired_payload_refs()` and `_find_affected_run_ids()` to include operation calls via UNION pattern
- [x] **server.py**: Updated `get_llm_usage_report()` to include operation LLM calls via UNION pattern
- [x] **exporter.py**: Added operations export with `get_operations_for_run()` method; call records now include `operation_id` field
- [x] **recorder.py**: Added `get_operations_for_run()` method for run-level operation queries
- [x] **lineage.py**: Unchanged - token-centric lineage is correct as-is; operation calls are at run/node level (different abstraction)

---

## Operational Health Monitoring

### Orphaned Operation Detection

Operations stuck in `open` status indicate framework bugs or resource exhaustion.
Add this to MCP `diagnose()` and consider a scheduled health check:

```sql
-- Find orphaned operations (open for > 1 hour)
SELECT
    operation_id,
    run_id,
    node_id,
    operation_type,
    started_at,
    julianday('now') - julianday(started_at) AS age_days
FROM operations
WHERE status = 'open'
  AND started_at < datetime('now', '-1 hour')
ORDER BY started_at;
```

**Interpretation:**
- `source_load` open for hours → Source iterator never exhausted (streaming OK, but verify)
- `sink_write` open for hours → Sink.write() hung or crashed without finally block executing
- Multiple orphans → Possible systemic issue (DB connection pool exhausted, etc.)

**Streaming Source Distinction:**

For streaming sources (Kafka, webhooks) that legitimately run for hours, distinguish "healthy long-running" from "stuck":

```sql
-- Truly stuck: open operation with NO recent calls
SELECT op.*
FROM operations op
WHERE op.status = 'open'
  AND op.operation_type = 'source_load'
  AND op.started_at < datetime('now', '-1 hour')
  AND NOT EXISTS (
      SELECT 1 FROM calls c
      WHERE c.operation_id = op.operation_id
        AND c.recorded_at > datetime('now', '-5 minutes')
  );

-- Healthy streaming: open but actively polling
SELECT op.*, MAX(c.recorded_at) as last_call
FROM operations op
LEFT JOIN calls c ON c.operation_id = op.operation_id
WHERE op.status = 'open'
  AND op.operation_type = 'source_load'
GROUP BY op.operation_id
HAVING MAX(c.recorded_at) > datetime('now', '-5 minutes');
```

### Exception Safety in Finally Blocks

The `complete_operation()` call in `finally` blocks must not mask the original exception:

```python
# CORRECT PATTERN - preserves original exception
source_start = time.perf_counter()
status: Literal["completed", "failed"] = "completed"  # Optimistic default
error_msg: str | None = None

try:
    source_iterator = config.source.load(ctx)
    for row_index, source_item in enumerate(source_iterator):
        # ... process rows ...
except Exception as e:
    status = "failed"
    error_msg = str(e)
    raise  # Re-raise BEFORE finally runs complete_operation
finally:
    duration_ms = (time.perf_counter() - source_start) * 1000
    try:
        recorder.complete_operation(
            operation_id=source_operation.operation_id,
            status=status,
            error=error_msg,
            duration_ms=duration_ms,
        )
    except Exception as db_error:
        # Log critically but don't mask original exception
        logger.critical(
            "Failed to complete operation - audit trail incomplete",
            operation_id=source_operation.operation_id,
            db_error=str(db_error),
        )
        # Don't raise - let original exception propagate
```

**Required test:**
```python
def test_complete_operation_db_failure_does_not_mask_original_exception():
    """
    If source.load() raises ValueError AND complete_operation() fails:
    - ValueError must propagate to caller
    - DB failure must be logged critically
    - Operation left in 'open' state (detectable anomaly)
    """
```

---

## Why This is the Right Fix

1. **Semantic clarity**: Rows = data, Operations = framework actions. No confusion.

2. **Query safety**: Existing queries for rows/tokens/states are UNCHANGED. No filtering needed.

3. **Explicit modeling**: Operations are discoverable - the table exists, the contract exists, the API exists.

4. **Proper FKs**: `node_id` correctly references the acting node with database-enforced integrity.

5. **Indexable**: Can efficiently query "all sink operations" or "operations for node X".

6. **Extensible**: Future operation types (aggregation flush, coalesce merge) slot in cleanly.

7. **No technical debt**: This is the model we'd want anyway. Building it now is free.

---

## Alternatives Rejected

### A. Operational Tokens with Negative row_index

**Rejected by architecture review:** Creates semantic overloading, hidden two-class system, query fragility.

### B. Nullable state_id in calls

**Rejected:** Creates two-class call system, no explicit parent for orphaned calls.

### C. Separate source_states table

**Rejected:** Fragments the model, but Operations table is similar - the difference is Operations explicitly models the DIFFERENT semantics rather than pretending sources are like transforms.

---

## Decision Record

**Decision:** Implement explicit `operations` table for source/sink I/O audit.

**Rationale:**
- Pre-release is the one chance to get the audit model right
- Schema changes are free when there are no users
- The alternative (operational tokens) was unanimously rejected by architecture, QA, Python, and systems reviews
- CLAUDE.md says "Make the fix right, not quick"

**Date:** 2026-01-31
**Status:** APPROVED FOR IMPLEMENTATION
