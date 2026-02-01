# Plugin-Level Pipelining Design

**Date:** 2026-01-26
**Status:** Approved with Changes
**Extends:** ADR-001 (Plugin-Level Concurrency)
**Author:** Architecture Team
**Reviewed By:** Architecture Critic, Python Engineering, Quality Engineering, Systems Thinking

---

## Executive Summary

This document describes **plugin-level pipelining** — infrastructure that enables transforms to process multiple rows concurrently while guaranteeing strict FIFO output ordering. Unlike orchestrator-level pipelining (rejected), this approach **honors ADR-001** by keeping concurrency at the plugin boundary.

### Key Decisions

| Aspect | Decision |
|--------|----------|
| **Architecture** | Orchestrator unchanged; plugins opt-in to batched processing |
| **Core Primitive** | `RowReorderBuffer` — accept rows out-of-order, release in FIFO order |
| **Integration** | `BatchTransformMixin` — easy opt-in for any transform |
| **Coordination** | `BatchingPluginProtocol` — standardized interface for observability |
| **FIFO Enforcement** | Per-plugin, using submission sequence numbers |
| **Cross-Plugin Limits** | Orchestrator-level in-flight awareness (optional) |
| **Determinism** | Fully preserved — orchestrator sees synchronous row-by-row processing |
| **ADR-001 Compliance** | Extended, not superseded |

### Expected Outcomes

- **3x+ throughput** for LLM-heavy pipelines (full pool utilization)
- **Strict FIFO ordering** preserved — outputs identical to sequential execution
- **Deterministic audit trail** — no orchestrator-level race conditions
- **Opt-in complexity** — only transforms that need batching use it
- **Simple recovery** — no new checkpoint complexity
- **Contained failures** — plugin bugs don't crash orchestrator

### Review Board Summary

| Reviewer | Verdict | Key Concern | Resolution |
|----------|---------|-------------|------------|
| Architecture | ✅ Approve | Defensive fallback violates CLAUDE.md | Fixed in Section 4.2 |
| Python Engineering | ⚠️ Request Changes | `notify_all()` causes thundering herd | Fixed in Section 3.1 |
| Quality Engineering | ⚠️ Request Changes | Missing E2E and stress tests | Added in Section 9 |
| Systems Thinking | ⚠️ Request Changes | Fragmentation risk, cross-plugin coordination | Added Sections 5 and 7 |

---

## 1. Problem Statement

### 1.1 Current Architecture (ADR-001)

ADR-001 established that concurrency lives at the **plugin boundary**, not the orchestrator:

```
Orchestrator: Row 1 → Transform → Sink
              Row 2 → Transform → Sink  (waits for Row 1)
              Row 3 → Transform → Sink  (waits for Row 2)

Transform internally: [Query 1, Query 2, Query 3] → Pool (concurrent)
```

This keeps the orchestrator simple and deterministic. Each row completes fully before the next starts.

### 1.2 The Performance Problem

With LLM transforms, each row issues ~10 queries. The `PooledExecutor` has `pool_size=30`, but only one row uses it at a time:

```
Row 1: [====== 10 queries ======]....................
Row 2:                           [====== 10 queries ======]....................
Row 3:                                                      [====== 10 queries ======]

Pool utilization: ~33% (10/30 slots used at any time)
```

**The pool is shared, but only one row uses it at a time because the orchestrator waits for each row to complete.**

### 1.3 The Insight

The existing `ReorderBuffer` already solves "process concurrently, release in order" for **queries within a row**. We need the same pattern for **rows within a plugin**.

```
Current (queries):    Row submits Q1, Q2, Q3 → Pool processes concurrently
                      ReorderBuffer releases in submission order

Proposed (rows):      Plugin accepts R1, R2, R3 → Process concurrently
                      RowReorderBuffer releases in submission order
```

### 1.4 Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| R1 | Transforms can process multiple rows concurrently | Must | ✅ Addressed |
| R2 | Output order matches input order (FIFO) | Must | ✅ Addressed |
| R3 | Orchestrator remains unchanged (sequential dispatch) | Must | ✅ Addressed |
| R4 | Audit trail remains deterministic | Must | ✅ Addressed |
| R5 | Opt-in per plugin (not global) | Must | ✅ Addressed |
| R6 | Shared query pool across rows | Must | ✅ Addressed |
| R7 | Backpressure when too many rows pending | Should | ✅ Addressed |
| R8 | No new checkpoint complexity | Should | ✅ Addressed |
| R9 | Cross-plugin coordination / limits | Should | ✅ Added (Section 7) |
| R10 | Standardized observability interface | Should | ✅ Added (Section 5) |

---

## 2. Architecture Overview

### 2.1 Conceptual Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PLUGIN-LEVEL PIPELINING                                  │
│                                                                              │
│  Orchestrator (UNCHANGED - sequential)                                       │
│       │                                                                      │
│       │  for row in source:                                                 │
│       │      result = transform.process(row)  # Blocks until FIFO-ready    │
│       │      sink.write(result)                                             │
│       │                                                                      │
│  ─────┼──────────────────────────────────────────────────────────────────── │
│       │                                                                      │
│       ▼  Inside BatchTransform                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   Row 1 ─► submit(seq=1) ─► [process async] ─► complete ─► RELEASE  │   │
│  │   Row 2 ─► submit(seq=2) ─► [process async] ─► complete ─► WAIT ────│──►│
│  │   Row 3 ─► submit(seq=3) ─► [process async] ─► complete ─► WAIT     │   │
│  │                                                                      │   │
│  │   RowReorderBuffer: releases seq=1, then seq=2, then seq=3          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  From orchestrator's view: synchronous call, returns in order               │
│  Inside plugin: concurrent processing with ordered release                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Insight: Blocking Semantics

When the orchestrator calls `transform.process(row)`:

1. The transform **submits** the row to its internal buffer (assigns sequence number)
2. The transform **starts** async processing (queries go to pool)
3. The transform **blocks** until THIS row is ready for release (FIFO order)
4. The transform **returns** the result

**Row 2's `process()` blocks not because Row 2 isn't done, but because Row 1 hasn't been released yet.**

This preserves the orchestrator's synchronous contract while enabling internal concurrency.

### 2.3 Component Overview

| Component | Purpose | Location |
|-----------|---------|----------|
| `RowReorderBuffer` | Accept rows, release in FIFO order | `src/elspeth/plugins/batching/reorder_buffer.py` |
| `BatchTransformMixin` | Easy integration for transforms | `src/elspeth/plugins/batching/mixin.py` |
| `BatchingPluginProtocol` | Standardized observability interface | `src/elspeth/plugins/batching/protocol.py` |
| `RowTicket` | Tracks a row through the buffer | `src/elspeth/plugins/batching/reorder_buffer.py` |

### 2.4 Package Structure

```
src/elspeth/plugins/batching/
├── __init__.py
├── reorder_buffer.py      # RowReorderBuffer, RowTicket
├── mixin.py               # BatchTransformMixin
├── protocol.py            # BatchingPluginProtocol
├── metrics.py             # BatchMetrics dataclass
└── coordination.py        # Global in-flight limiter (optional)
```

---

## 3. Core Component: RowReorderBuffer

### 3.1 Design

`RowReorderBuffer` is the row-level equivalent of `ReorderBuffer`. It:

- Assigns monotonic sequence numbers on submission
- Tracks completion status for each row
- Releases rows in strict submission order
- Provides backpressure when too many rows are pending

**Threading Model (from Python Engineering Review):**
- Single `threading.Lock` protects all state
- Two `Condition` variables: `_submit_condition` (backpressure), `_release_condition` (FIFO)
- Uses `notify()` (not `notify_all()`) to avoid thundering herd

```python
from dataclasses import dataclass
from typing import Any
import threading
import time


@dataclass(frozen=True)
class RowTicket:
    """Handle for a row submitted to the reorder buffer."""
    sequence: int
    row_id: str
    submitted_at: float


class RowReorderBuffer:
    """
    Accept rows out-of-order, release in submission order.

    Thread-safe. Designed for transforms that process multiple rows
    concurrently but must maintain FIFO output ordering.

    This is the row-level equivalent of ReorderBuffer (used for queries).

    Thread Safety Model:
        - submit(): Called by orchestrator thread, may block on backpressure
        - complete(): Called by worker threads, wakes release waiters
        - wait_for_release(): Called by orchestrator thread, blocks until FIFO-ready

    Invariants:
        - next_release_seq <= min(in_flight)
        - completed.keys() ⊆ in_flight
        - len(in_flight) <= max_pending
    """

    def __init__(
        self,
        max_pending: int = 100,
        name: str = "row-reorder",
    ):
        """
        Args:
            max_pending: Maximum rows that can be pending (backpressure threshold)
            name: Name for logging/metrics
        """
        self._name = name
        self._max_pending = max_pending

        self._lock = threading.Lock()
        self._submit_condition = threading.Condition(self._lock)
        self._release_condition = threading.Condition(self._lock)

        # Sequence tracking
        self._next_submit_seq = 0
        self._next_release_seq = 0

        # Completed rows waiting for release: seq -> result
        self._completed: dict[int, Any] = {}

        # In-flight tracking for metrics
        self._in_flight: set[int] = set()

        # Shutdown flag (use Event for idiomatic shutdown signaling)
        self._shutdown = threading.Event()

        # Metrics
        self._total_submitted = 0
        self._total_released = 0
        self._max_wait_time_ms: float = 0.0
        self._total_wait_time_ms: float = 0.0

    def submit(self, row_id: str) -> RowTicket:
        """
        Submit a row for processing. Returns a ticket to complete later.

        Blocks if max_pending rows are already in flight (backpressure).
        Does NOT use timeout polling - relies on proper notification.

        Args:
            row_id: Unique identifier for the row

        Returns:
            RowTicket to pass to complete() when processing finishes

        Raises:
            RuntimeError: If buffer is shut down
        """
        with self._lock:
            # Backpressure: wait if too many pending
            while len(self._in_flight) >= self._max_pending:
                if self._shutdown.is_set():
                    raise RuntimeError(f"Buffer '{self._name}' is shut down")
                # No timeout - wake on notify only (fixes thundering herd)
                self._submit_condition.wait()

            seq = self._next_submit_seq
            self._next_submit_seq += 1
            self._in_flight.add(seq)
            self._total_submitted += 1

            return RowTicket(
                sequence=seq,
                row_id=row_id,
                submitted_at=time.time(),
            )

    def complete(self, ticket: RowTicket, result: Any) -> None:
        """
        Mark a row as complete. It will be released when predecessors are done.

        Args:
            ticket: The ticket from submit()
            result: The processing result to return

        Raises:
            ValueError: If ticket already completed
        """
        with self._lock:
            if ticket.sequence in self._completed:
                raise ValueError(
                    f"Ticket {ticket.sequence} (row_id={ticket.row_id}) already completed"
                )

            self._completed[ticket.sequence] = result
            # Use notify() not notify_all() - only one waiter can be next in sequence
            self._release_condition.notify()

    def wait_for_release(self, ticket: RowTicket, timeout: float | None = None) -> Any:
        """
        Block until this ticket's row is ready for release.

        Returns the result when all predecessors have been released.

        Args:
            ticket: The ticket from submit()
            timeout: Maximum seconds to wait (None = forever)

        Returns:
            The result passed to complete()

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If buffer is shut down
        """
        deadline = time.time() + timeout if timeout else None
        wait_start = time.time()

        with self._lock:
            while True:
                # Check shutdown first
                if self._shutdown.is_set():
                    raise RuntimeError(f"Buffer '{self._name}' is shut down")

                # Check if we're next in sequence AND completed
                if ticket.sequence == self._next_release_seq:
                    if ticket.sequence in self._completed:
                        result = self._completed.pop(ticket.sequence)
                        self._in_flight.discard(ticket.sequence)
                        self._next_release_seq += 1
                        self._total_released += 1

                        # Track wait time metrics
                        wait_time_ms = (time.time() - wait_start) * 1000
                        self._total_wait_time_ms += wait_time_ms
                        self._max_wait_time_ms = max(self._max_wait_time_ms, wait_time_ms)

                        # Wake one submitter (backpressure relief)
                        self._submit_condition.notify()
                        # Wake next release waiter
                        self._release_condition.notify()

                        return result

                # Not ready yet - wait
                if deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Timeout waiting for ticket {ticket.sequence} "
                            f"(row_id={ticket.row_id}, next_release={self._next_release_seq})"
                        )
                    self._release_condition.wait(timeout=remaining)
                else:
                    # No timeout - wait for notification
                    self._release_condition.wait()

    def shutdown(self) -> None:
        """Signal shutdown. Wakes all waiters with RuntimeError."""
        self._shutdown.set()
        with self._lock:
            self._submit_condition.notify_all()  # Wake ALL for shutdown
            self._release_condition.notify_all()  # Wake ALL for shutdown

    # --- Metrics (implements BatchingPluginProtocol) ---

    @property
    def pending_count(self) -> int:
        """Number of rows in flight (submitted but not released)."""
        with self._lock:
            return len(self._in_flight)

    @property
    def completed_waiting_count(self) -> int:
        """Number of completed rows waiting for predecessors."""
        with self._lock:
            return len(self._completed)

    @property
    def next_release_seq(self) -> int:
        """Next sequence number to be released."""
        with self._lock:
            return self._next_release_seq

    def get_metrics(self) -> "BatchMetrics":
        """Get comprehensive metrics snapshot."""
        with self._lock:
            return BatchMetrics(
                name=self._name,
                max_pending=self._max_pending,
                current_pending=len(self._in_flight),
                current_waiting=len(self._completed),
                total_submitted=self._total_submitted,
                total_released=self._total_released,
                max_wait_time_ms=self._max_wait_time_ms,
                avg_wait_time_ms=(
                    self._total_wait_time_ms / self._total_released
                    if self._total_released > 0
                    else 0.0
                ),
            )
```

### 3.2 Thread Safety Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     THREAD SAFETY MODEL                                      │
│                                                                              │
│  Orchestrator Thread:                                                        │
│      submit() ──► returns ticket immediately (or blocks on backpressure)    │
│      wait_for_release() ──► blocks until FIFO-ready                         │
│                                                                              │
│  Worker Thread 1:                                                            │
│      complete() ──► marks done, calls notify() to wake release waiter       │
│                                                                              │
│  Worker Thread 2:                                                            │
│      complete() ──► marks done, calls notify() to wake release waiter       │
│                                                                              │
│  Synchronization:                                                            │
│      - Single lock protects all state                                        │
│      - submit_condition: for backpressure (wake ONE on slot available)      │
│      - release_condition: for FIFO ordering (wake ONE on predecessor done)  │
│                                                                              │
│  Key Fix (from Python Review):                                               │
│      - Use notify() not notify_all() to avoid thundering herd               │
│      - With 30 pending rows, notify_all() wakes 30 threads; 29 sleep again  │
│      - notify() wakes only the ONE thread that can proceed                  │
│                                                                              │
│  Exception: shutdown() uses notify_all() to wake ALL waiters for cleanup    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Comparison with ReorderBuffer

| Aspect | ReorderBuffer (queries) | RowReorderBuffer (rows) |
|--------|------------------------|------------------------|
| **Use case** | Queries within one row | Rows within one transform |
| **Caller** | PooledExecutor | BatchTransformMixin |
| **Blocking** | `get_ready_results()` polling | `wait_for_release()` blocking |
| **Backpressure** | Via semaphore in pool | Built-in max_pending |
| **Thread model** | Single consumer | Multiple blocked callers |
| **Wakeup pattern** | N/A (polling) | `notify()` (single waiter) |

---

## 4. Integration: BatchTransformMixin

### 4.1 Design

`BatchTransformMixin` provides easy opt-in for any transform:

```python
from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable
import traceback

from elspeth.plugins.batching.reorder_buffer import RowReorderBuffer, RowTicket
from elspeth.plugins.batching.protocol import BatchingPluginProtocol
from elspeth.plugins.batching.metrics import BatchMetrics
from elspeth.core.context import PluginContext
from elspeth.core.results import TransformResult


class BatchTransformMixin(BatchingPluginProtocol):
    """
    Mixin that adds batch processing capability to any transform.

    Transforms using this mixin can process multiple rows concurrently
    while guaranteeing FIFO output order. The orchestrator sees
    synchronous behavior; concurrency is hidden inside the plugin.

    Implements BatchingPluginProtocol for standardized observability.

    Usage:
        class MyLLMTransform(BaseTransform, BatchTransformMixin):
            def __init__(self, config, *, shared_pool=None):
                super().__init__(config)
                self.init_batch_processing(max_pending=20)
                self._pool = shared_pool or PooledExecutor(...)

            def process(self, row, ctx):
                # Called by orchestrator for each row
                return self.process_batched(row, ctx, self._do_processing)

            def _do_processing(self, row, ctx):
                # Actual processing logic (runs concurrently)
                result = self._pool.execute(self._make_query(row))
                return TransformResult.success({"output": result})
    """

    _batch_buffer: RowReorderBuffer
    _batch_executor: ThreadPoolExecutor
    _batch_futures: dict[int, Future[None]]  # Type hint (from Python Review)

    def init_batch_processing(
        self,
        max_pending: int = 20,
        max_workers: int | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialize batch processing infrastructure.

        Call this in __init__ after super().__init__().

        IMPORTANT: max_workers should equal max_pending to avoid thread starvation.
        Each pending row needs a worker thread to process it.

        Args:
            max_pending: Max rows that can be pending (backpressure)
            max_workers: Worker threads for async processing (default: max_pending)
            name: Name for metrics/logging (default: class name)
        """
        buffer_name = name or self.__class__.__name__
        self._batch_buffer = RowReorderBuffer(
            max_pending=max_pending,
            name=buffer_name,
        )
        # max_workers MUST equal max_pending (Architecture Review recommendation)
        self._batch_executor = ThreadPoolExecutor(
            max_workers=max_workers or max_pending,
            thread_name_prefix=f"{buffer_name}-batch",
        )
        self._batch_futures = {}

    def process_batched(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
        processor: Callable[[dict[str, Any], PluginContext], TransformResult],
    ) -> TransformResult:
        """
        Process a row with batching.

        This is the main entry point. Call this from process().

        Args:
            row: The row data
            ctx: Plugin context (MUST have ctx.token set)
            processor: Function that does the actual processing

        Returns:
            TransformResult (in FIFO order)

        Raises:
            ValueError: If ctx.token is None (required for row_id)
        """
        # No defensive fallback - ctx.token is required (CLAUDE.md compliance)
        if ctx.token is None:
            raise ValueError(
                f"BatchTransformMixin requires ctx.token to be set. "
                f"This is a bug in the calling code."
            )
        row_id = ctx.token.token_id

        # 1. Submit to buffer (assigns sequence, may block on backpressure)
        ticket = self._batch_buffer.submit(row_id)

        # 2. Start async processing
        future = self._batch_executor.submit(
            self._process_and_complete,
            ticket,
            row,
            ctx,
            processor,
        )
        self._batch_futures[ticket.sequence] = future

        # 3. Wait for THIS row to be released (blocks until predecessors done)
        try:
            result = self._batch_buffer.wait_for_release(ticket)
            return result
        finally:
            # Cleanup future reference
            self._batch_futures.pop(ticket.sequence, None)

    def _process_and_complete(
        self,
        ticket: RowTicket,
        row: dict[str, Any],
        ctx: PluginContext,
        processor: Callable[[dict[str, Any], PluginContext], TransformResult],
    ) -> None:
        """Worker thread: process row and mark complete."""
        try:
            result = processor(row, ctx)
            self._batch_buffer.complete(ticket, result)
        except Exception as e:
            # Capture full traceback for debugging (Python Review recommendation)
            tb = traceback.format_exc()
            self._batch_buffer.complete(
                ticket,
                TransformResult.error(
                    row,
                    {
                        "error": str(e),
                        "type": type(e).__name__,
                        "traceback": tb,
                    },
                ),
            )

    def shutdown_batch_processing(self) -> None:
        """Shutdown batch processing. Call in cleanup()."""
        self._batch_buffer.shutdown()
        self._batch_executor.shutdown(wait=True)

    # --- BatchingPluginProtocol implementation ---

    def get_in_flight_count(self) -> int:
        """Number of rows currently in flight."""
        return self._batch_buffer.pending_count

    def get_batch_metrics(self) -> BatchMetrics:
        """Get comprehensive batch metrics."""
        return self._batch_buffer.get_metrics()

    def shutdown_gracefully(self, timeout: float) -> None:
        """Shutdown with timeout."""
        self._batch_buffer.shutdown()
        self._batch_executor.shutdown(wait=True)
```

### 4.2 Key Design Decisions

**No Defensive Fallbacks (CLAUDE.md Compliance):**

```python
# WRONG - defensive programming that hides bugs
row_id = ctx.token.token_id if ctx.token else str(id(row))

# RIGHT - fail fast if invariant violated
if ctx.token is None:
    raise ValueError("BatchTransformMixin requires ctx.token to be set")
row_id = ctx.token.token_id
```

**Traceback Capture (Python Review):**

```python
# WRONG - loses debugging information
except Exception as e:
    self._batch_buffer.complete(ticket, TransformResult.error(row, {"error": str(e)}))

# RIGHT - preserves full traceback for debugging
except Exception as e:
    tb = traceback.format_exc()
    self._batch_buffer.complete(
        ticket,
        TransformResult.error(row, {"error": str(e), "type": type(e).__name__, "traceback": tb}),
    )
```

---

## 5. Standardized Interface: BatchingPluginProtocol

**(Added based on Systems Thinking Review - prevents fragmentation)**

### 5.1 Protocol Definition

```python
from typing import Protocol
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchMetrics:
    """Metrics snapshot from a batching plugin."""
    name: str
    max_pending: int
    current_pending: int
    current_waiting: int  # Completed but waiting for FIFO release
    total_submitted: int
    total_released: int
    max_wait_time_ms: float
    avg_wait_time_ms: float


class BatchingPluginProtocol(Protocol):
    """
    Contract for plugins using batch processing.

    All batching plugins MUST implement this protocol to enable:
    - Observability (metrics dashboards)
    - Cross-plugin coordination (global in-flight limits)
    - Graceful shutdown
    - Debugging (introspection of buffer state)

    This protocol prevents fragmentation by standardizing the interface
    across all batching plugins (Systems Thinking Review recommendation).
    """

    def get_in_flight_count(self) -> int:
        """Return number of rows currently in flight in this plugin."""
        ...

    def get_batch_metrics(self) -> BatchMetrics:
        """Return comprehensive metrics snapshot."""
        ...

    def shutdown_gracefully(self, timeout: float) -> None:
        """Shutdown batch processing, waiting up to timeout seconds."""
        ...
```

### 5.2 Why This Matters

Without a standardized protocol, each plugin would implement batching differently:
- Different metric names
- Different shutdown behavior
- No way for orchestrator to introspect total in-flight rows

The protocol enables:
1. **Observability**: Unified metrics across all batching plugins
2. **Coordination**: Orchestrator can sum `get_in_flight_count()` across plugins
3. **Graceful shutdown**: Consistent shutdown behavior
4. **Testing**: Standard interface for mocking

---

## 6. Usage Example: LLM Transform

### 6.1 Converting an Existing Transform

```python
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.batching import BatchTransformMixin
from elspeth.plugins.pooling import PooledExecutor
from elspeth.core.context import PluginContext
from elspeth.core.results import TransformResult


class AzureMultiQueryLLMTransform(BaseTransform, BatchTransformMixin):
    """
    LLM transform with row-level batching.

    Multiple rows are processed concurrently (queries submitted to shared pool),
    but output is released in strict FIFO order.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        shared_pool: PooledExecutor | None = None,
    ):
        super().__init__(config)

        # Initialize batch processing to match pool depth
        pool_size = config.get("pool_size", 30)
        self.init_batch_processing(
            max_pending=pool_size,
            name="azure-multi-query-llm",
        )

        # Shared pool for queries (across all rows)
        self._pool = shared_pool or PooledExecutor(
            pool_size=pool_size,
            # ... other config
        )

    def process(self, row: dict, ctx: PluginContext) -> TransformResult:
        """Called by orchestrator - uses batched processing."""
        return self.process_batched(row, ctx, self._do_llm_processing)

    def _do_llm_processing(self, row: dict, ctx: PluginContext) -> TransformResult:
        """
        Actual LLM processing for one row. Runs concurrently.

        This is where queries go to the shared pool.
        """
        try:
            # Build queries for this row
            queries = self._build_queries(row)

            # Submit to shared pool (concurrent with other rows' queries)
            results = self._pool.execute_batch(queries)

            # Process results
            output = self._merge_results(row, results)

            return TransformResult.success(output)

        except Exception as e:
            return TransformResult.error(row, {"error": str(e)})

    def cleanup(self, ctx: PluginContext) -> None:
        """Cleanup on pipeline end."""
        self.shutdown_batch_processing()
        super().cleanup(ctx)
```

### 6.2 Initial Adoption Scope

**(Systems Thinking Review recommendation: limit initial adoption)**

**Phase 1 (RC-1):** LLM transforms only
- `AzureMultiQueryLLMTransform`
- `OpenRouterMultiQueryTransform`

**Phase 2 (Post RC-1):** Expand after validating
- HTTP API transforms
- ML inference transforms
- Other I/O-bound transforms

```yaml
# settings.yaml - Phase 1 validation
transforms:
  - name: llm_classifier
    type: azure_multi_query_llm
    config:
      batch_processing:
        enabled: true           # ✅ Allowed in Phase 1
        max_pending: 30

  - name: custom_transform
    type: my_custom_transform
    config:
      batch_processing:
        enabled: true           # ❌ Not yet supported - will warn/error
```

---

## 7. Cross-Plugin Coordination

**(Added based on Systems Thinking Review - addresses R3 feedback loop)**

### 7.1 The Problem

When multiple plugins batch concurrently:

```
Transform A: 30 rows in buffer
Transform B: 30 rows in buffer
Transform C: 30 rows in buffer
─────────────────────────────────
Total: 90 rows in-flight with NO orchestrator awareness
```

This can cause:
- Memory exhaustion
- Cascading backpressure
- Resource contention

### 7.2 Solution: Global In-Flight Limiter (Optional)

```python
# src/elspeth/plugins/batching/coordination.py

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.plugins.batching.protocol import BatchingPluginProtocol


class GlobalInFlightLimiter:
    """
    Orchestrator-level awareness of total in-flight rows across all batching plugins.

    This is OPTIONAL - pipelines without coordination still work correctly,
    but may experience higher memory usage with multiple batching plugins.

    Usage:
        limiter = GlobalInFlightLimiter(max_total=100)

        # Register each batching plugin
        limiter.register(transform_a)
        limiter.register(transform_b)

        # Check before allowing new work (called by orchestrator)
        limiter.wait_for_capacity()  # Blocks if total >= max_total
    """

    def __init__(self, max_total: int = 100):
        self._max_total = max_total
        self._plugins: list[BatchingPluginProtocol] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def register(self, plugin: "BatchingPluginProtocol") -> None:
        """Register a batching plugin for coordination."""
        with self._lock:
            self._plugins.append(plugin)

    def get_total_in_flight(self) -> int:
        """Get total rows in flight across all registered plugins."""
        with self._lock:
            return sum(p.get_in_flight_count() for p in self._plugins)

    def wait_for_capacity(self, timeout: float | None = None) -> None:
        """
        Block until total in-flight is below max_total.

        Called by orchestrator before processing each row.
        """
        deadline = time.time() + timeout if timeout else None

        with self._lock:
            while self.get_total_in_flight() >= self._max_total:
                if deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError("Timeout waiting for global capacity")
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait(timeout=1.0)

    def notify_released(self) -> None:
        """Called when a row is released from any plugin."""
        with self._lock:
            self._condition.notify()
```

### 7.3 Configuration

```yaml
# settings.yaml
concurrency:
  # Per-plugin batching (existing)
  pool_size: 30

  # Cross-plugin coordination (NEW - optional)
  global_coordination:
    enabled: true              # Enable orchestrator awareness
    max_total_in_flight: 100   # Max rows across ALL batching plugins
```

---

## 8. Crash Recovery Behavior

**(Added based on Systems Thinking Review - documents implicit behavior)**

### 8.1 What Happens on Crash

If the orchestrator crashes with rows in plugin buffers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CRASH RECOVERY BEHAVIOR                                  │
│                                                                              │
│  At crash time:                                                              │
│    - Rows 1-40: Released to sink (checkpointed)                             │
│    - Rows 41-70: In plugin buffer (NOT checkpointed - in-memory only)       │
│    - Rows 71+: Not yet pulled from source                                   │
│                                                                              │
│  On resume:                                                                  │
│    1. Orchestrator loads last checkpoint: "released through row 40"         │
│    2. Source resumes from row 41                                            │
│    3. Rows 41-70 are RE-PROCESSED (they were never checkpointed)            │
│                                                                              │
│  Key insight: RowReorderBuffer is IN-MEMORY ONLY.                           │
│               It does NOT persist state to Landscape.                        │
│               This is intentional - keeps checkpoint model simple.           │
│                                                                              │
│  Trade-off:                                                                  │
│    - Pro: No new checkpoint complexity (R8 requirement)                     │
│    - Con: Up to max_pending rows may be re-processed on crash               │
│                                                                              │
│  For LLM transforms, re-processing is acceptable:                           │
│    - LLM calls are idempotent (same input → same output)                   │
│    - Cost is bounded (max_pending × cost_per_row)                          │
│    - No duplicate sink writes (FIFO release ensures order)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Comparison with Aggregation Crash Recovery

| Aspect | Aggregation Batches | RowReorderBuffer |
|--------|--------------------|--------------------|
| State persistence | `draft_batches` in Landscape | In-memory only |
| Recovery behavior | Resume from checkpoint | Re-process in-flight rows |
| Duplicate protection | `batch_id` idempotency | FIFO release order |
| Complexity | Higher (stateful) | Lower (stateless) |

### 8.3 Design Rationale

We chose in-memory buffers because:
1. **Checkpoint simplicity** (R8) - No new Landscape tables
2. **Bounded re-work** - At most `max_pending` rows re-processed
3. **Idempotent transforms** - LLM calls give same results on retry
4. **No sink duplicates** - FIFO release ensures order even after crash

---

## 9. Testing Strategy

**(Significantly expanded based on Quality Engineering Review)**

### 9.1 Test Categories

| Category | Purpose | Count | Priority |
|----------|---------|-------|----------|
| Unit (RowReorderBuffer) | Component correctness | ~20 | P0 |
| Unit (BatchTransformMixin) | Integration correctness | ~10 | P0 |
| Stress (1000+ rows) | Lock contention, race conditions | ~5 | P0 |
| Deadlock | Shutdown safety, error recovery | ~5 | P0 |
| Property-based | FIFO invariant across random scenarios | ~5 | P0 |
| E2E (Orchestrator) | Full pipeline with Landscape | ~5 | P0 |
| Performance | Benchmark 3x throughput claim | ~3 | P1 |

### 9.2 P0 Critical Tests (Must Pass Before Merge)

#### 9.2.1 Unit Tests for RowReorderBuffer

```python
class TestRowReorderBuffer:
    """Unit tests for RowReorderBuffer."""

    def test_fifo_ordering_sequential_complete(self):
        """Rows completed in order are released in order."""
        buffer = RowReorderBuffer(max_pending=10)

        t1 = buffer.submit("row-1")
        t2 = buffer.submit("row-2")
        t3 = buffer.submit("row-3")

        # Complete in order
        buffer.complete(t1, "result-1")
        buffer.complete(t2, "result-2")
        buffer.complete(t3, "result-3")

        # Release in order
        assert buffer.wait_for_release(t1) == "result-1"
        assert buffer.wait_for_release(t2) == "result-2"
        assert buffer.wait_for_release(t3) == "result-3"

    def test_fifo_ordering_reverse_complete(self):
        """Rows completed in reverse order still release in submission order."""
        buffer = RowReorderBuffer(max_pending=10)

        t1 = buffer.submit("row-1")
        t2 = buffer.submit("row-2")
        t3 = buffer.submit("row-3")

        # Complete in REVERSE order
        buffer.complete(t3, "result-3")
        buffer.complete(t2, "result-2")
        buffer.complete(t1, "result-1")

        # Still release in submission order
        assert buffer.wait_for_release(t1) == "result-1"
        assert buffer.wait_for_release(t2) == "result-2"
        assert buffer.wait_for_release(t3) == "result-3"

    def test_backpressure_blocks_submit(self):
        """Submit blocks when max_pending reached."""
        buffer = RowReorderBuffer(max_pending=2)

        t1 = buffer.submit("row-1")
        t2 = buffer.submit("row-2")

        # Third submit should block
        submit_completed = threading.Event()

        def try_submit():
            buffer.submit("row-3")
            submit_completed.set()

        thread = threading.Thread(target=try_submit)
        thread.start()

        # Should not complete immediately
        assert not submit_completed.wait(timeout=0.1)

        # Release one row
        buffer.complete(t1, "result-1")
        buffer.wait_for_release(t1)

        # Now submit should complete
        assert submit_completed.wait(timeout=1.0)
        thread.join()

    def test_metrics_accurate(self):
        """Metrics reflect actual buffer state."""
        buffer = RowReorderBuffer(max_pending=10)

        assert buffer.pending_count == 0
        assert buffer.completed_waiting_count == 0

        t1 = buffer.submit("row-1")
        assert buffer.pending_count == 1

        t2 = buffer.submit("row-2")
        assert buffer.pending_count == 2

        buffer.complete(t2, "result-2")  # Complete out of order
        assert buffer.completed_waiting_count == 1

        buffer.complete(t1, "result-1")
        assert buffer.completed_waiting_count == 2

        buffer.wait_for_release(t1)
        assert buffer.pending_count == 1
        assert buffer.completed_waiting_count == 1
```

#### 9.2.2 Deadlock Tests (Critical)

```python
class TestDeadlockScenarios:
    """Tests for deadlock and shutdown safety."""

    def test_shutdown_during_wait_releases_all_waiters(self):
        """Shutdown during wait should wake all waiting threads, not deadlock."""
        buffer = RowReorderBuffer(max_pending=10)

        # Submit 5 tickets
        tickets = [buffer.submit(f"row-{i}") for i in range(5)]

        # Complete only the LAST ticket (0-3 will block waiting)
        buffer.complete(tickets[4], "result-4")

        # Start threads waiting for tickets 0-3
        exceptions = [None] * 4

        def wait_for_ticket(idx):
            try:
                buffer.wait_for_release(tickets[idx], timeout=5.0)
            except Exception as e:
                exceptions[idx] = e

        wait_threads = []
        for i in range(4):
            t = threading.Thread(target=wait_for_ticket, args=(i,))
            wait_threads.append(t)
            t.start()

        time.sleep(0.1)  # Let threads start waiting

        # Now shutdown - should wake all waiters
        buffer.shutdown()

        for t in wait_threads:
            t.join(timeout=2.0)
            assert not t.is_alive(), "Thread deadlocked after shutdown"

        # All waiters should get RuntimeError (shutdown)
        for i in range(4):
            assert isinstance(exceptions[i], RuntimeError)
            assert "shut down" in str(exceptions[i])

    def test_double_complete_raises_but_does_not_deadlock_waiters(self):
        """Double-complete should raise ValueError but not leave waiters hanging."""
        buffer = RowReorderBuffer(max_pending=10)

        t0 = buffer.submit("row-0")
        t1 = buffer.submit("row-1")

        # Complete t1 first (t0 blocks)
        buffer.complete(t1, "result-1")

        # Start waiter for t0
        wait_result = [None]
        wait_exception = [None]

        def wait_for_t0():
            try:
                wait_result[0] = buffer.wait_for_release(t0, timeout=5.0)
            except Exception as e:
                wait_exception[0] = e

        wait_thread = threading.Thread(target=wait_for_t0)
        wait_thread.start()

        time.sleep(0.05)  # Let waiter start

        # Try to double-complete t1 (should raise)
        with pytest.raises(ValueError, match="already completed"):
            buffer.complete(t1, "result-1-again")

        # Now properly complete t0
        buffer.complete(t0, "result-0")

        wait_thread.join(timeout=2.0)
        assert not wait_thread.is_alive(), "Waiter deadlocked after double-complete"

        assert wait_result[0] == "result-0"
```

#### 9.2.3 Stress Tests (1000+ Rows)

```python
class TestStress:
    """Stress tests for race conditions under heavy load."""

    def test_sustained_concurrent_load_1000_rows(self):
        """Stress test: 1000 rows submitted and released correctly."""
        buffer = RowReorderBuffer(max_pending=100)
        num_rows = 1000
        num_threads = 10

        tickets = [None] * num_rows
        results = [None] * num_rows

        # Submit all tickets (from main thread, sequentially)
        for i in range(num_rows):
            tickets[i] = buffer.submit(f"row-{i}")

        # Complete in random order from multiple threads
        def completer(indices):
            import random
            shuffled = indices.copy()
            random.shuffle(shuffled)
            for i in shuffled:
                time.sleep(random.uniform(0.0001, 0.001))
                buffer.complete(tickets[i], f"result-{i}")

        complete_threads = []
        indices = list(range(num_rows))
        for t in range(num_threads):
            chunk = indices[t::num_threads]
            thread = threading.Thread(target=completer, args=(chunk,))
            complete_threads.append(thread)
            thread.start()

        # Release and verify order (from main thread)
        for i in range(num_rows):
            results[i] = buffer.wait_for_release(tickets[i], timeout=30.0)

        for t in complete_threads:
            t.join()

        # Verify FIFO order maintained
        assert results == [f"result-{i}" for i in range(num_rows)]

    def test_no_race_in_metrics_under_concurrent_load(self):
        """Metrics should be accurate even with concurrent operations."""
        buffer = RowReorderBuffer(max_pending=50)
        num_rows = 500

        # Track max observed pending
        max_observed_pending = [0]

        def monitor():
            for _ in range(100):
                pending = buffer.pending_count
                max_observed_pending[0] = max(max_observed_pending[0], pending)
                time.sleep(0.01)

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        # Process rows
        tickets = []
        for i in range(num_rows):
            t = buffer.submit(f"row-{i}")
            tickets.append(t)
            buffer.complete(t, f"result-{i}")
            buffer.wait_for_release(t)

        monitor_thread.join()

        # Verify metrics never exceeded max
        assert max_observed_pending[0] <= 50
```

#### 9.2.4 Property-Based Tests

```python
from hypothesis import given, strategies as st, settings

class TestFIFOProperties:
    """Property-based tests for FIFO invariant."""

    @given(
        num_rows=st.integers(min_value=1, max_value=100),
        max_pending=st.integers(min_value=1, max_value=50),
        delays=st.lists(st.floats(min_value=0.0001, max_value=0.01), min_size=1, max_size=100),
    )
    @settings(max_examples=50, deadline=60000)
    def test_fifo_never_violated(self, num_rows, max_pending, delays):
        """PROPERTY: Release order always equals submission order."""
        buffer = RowReorderBuffer(max_pending=min(max_pending, num_rows))

        tickets = [buffer.submit(f"row-{i}") for i in range(num_rows)]

        # Complete in random order with random delays
        def complete_with_delay(ticket, delay):
            time.sleep(delay)
            buffer.complete(ticket, ticket.sequence)

        threads = []
        for i, ticket in enumerate(tickets):
            delay = delays[i % len(delays)]
            t = threading.Thread(target=complete_with_delay, args=(ticket, delay))
            threads.append(t)
            t.start()

        # Wait for releases in submission order
        release_order = []
        for ticket in tickets:
            result = buffer.wait_for_release(ticket, timeout=10.0)
            release_order.append(result)

        for t in threads:
            t.join()

        # INVARIANT: release order == submission order
        assert release_order == list(range(num_rows))
```

#### 9.2.5 E2E Tests with Orchestrator

```python
class TestEndToEnd:
    """End-to-end tests with real Orchestrator and Landscape."""

    def test_batch_transform_audit_trail_deterministic(self, orchestrator, landscape):
        """
        Full pipeline: Verify Landscape records rows in source order.

        This is the CRITICAL test that validates audit trail determinism.
        """
        # Create pipeline with batch transform
        config = PipelineConfig(
            source=CSVSource("test_data.csv"),
            transforms=[
                BatchTestTransform(max_pending=10),
            ],
            sinks={"output": CSVSink("output.csv")},
        )

        # Run pipeline
        result = orchestrator.run(config)

        # Query Landscape for node_states
        node_states = landscape.get_node_states(result.run_id)

        # Verify node_states are in source order
        state_sequences = [
            (s.token_id, s.started_at, s.completed_at)
            for s in node_states
            if s.node_id == "batch_test_transform"
        ]

        # started_at should be in increasing order (orchestrator calls sequentially)
        started_times = [s[1] for s in state_sequences]
        assert started_times == sorted(started_times), "started_at not in order"

        # completed_at should ALSO be in order (FIFO release)
        completed_times = [s[2] for s in state_sequences]
        assert completed_times == sorted(completed_times), "completed_at not in order"

    def test_batch_transform_output_matches_sequential(self, orchestrator):
        """
        Batch transform should produce identical output to sequential.

        Run same pipeline with batching disabled, compare results.
        """
        source_data = [{"id": i, "value": f"test-{i}"} for i in range(100)]

        # Run with batching
        batch_result = run_pipeline(source_data, batch_enabled=True, max_pending=20)

        # Run without batching (sequential)
        sequential_result = run_pipeline(source_data, batch_enabled=False)

        # Results must be identical
        assert batch_result.output_rows == sequential_result.output_rows
```

### 9.3 P1 Performance Tests

```python
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks to validate throughput claims."""

    def test_batch_processing_achieves_3x_throughput(self, benchmark):
        """Batch processing should achieve at least 2.5x throughput vs sequential."""
        source_rows = 100
        simulated_query_time = 0.1  # 100ms per query
        queries_per_row = 10
        pool_size = 30

        # Sequential baseline
        sequential_time = source_rows * queries_per_row * simulated_query_time

        # Batch processing (pool fully utilized)
        # With 30 pool slots, 10 queries/row, 3 rows can process concurrently
        # Expected time: (source_rows / 3) * (queries_per_row * simulated_query_time)
        expected_batch_time = (source_rows / 3) * (queries_per_row * simulated_query_time / 10)

        # Actually run benchmark
        batch_time = benchmark(run_batch_pipeline, source_rows, pool_size)

        # Assert at least 2.5x improvement
        improvement = sequential_time / batch_time
        assert improvement >= 2.5, f"Only achieved {improvement:.1f}x improvement"
```

---

## 10. Configuration

### 10.1 Per-Transform Configuration

```yaml
# settings.yaml
transforms:
  - name: llm_classifier
    type: azure_multi_query_llm
    config:
      # ... existing LLM config ...

      # Batch processing settings (NEW)
      batch_processing:
        enabled: true           # Enable row batching
        max_pending: 30         # Max rows in flight (default: pool_size)
        release_timeout_seconds: 300.0  # Max wait for FIFO release
```

### 10.2 Global Coordination (Optional)

```yaml
# settings.yaml
concurrency:
  # Per-plugin batching
  pool_size: 30

  # Cross-plugin coordination (NEW - optional)
  global_coordination:
    enabled: true              # Enable orchestrator awareness
    max_total_in_flight: 100   # Max rows across ALL batching plugins
```

### 10.3 Pydantic Models

```python
class BatchProcessingSettings(BaseModel):
    """Configuration for plugin-level batch processing."""

    enabled: bool = Field(
        default=True,
        description="Enable row-level batching in this transform",
    )

    max_pending: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Max rows pending (default: match pool_size)",
    )

    release_timeout_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Max time to wait for FIFO release",
    )


class GlobalCoordinationSettings(BaseModel):
    """Configuration for cross-plugin coordination."""

    enabled: bool = Field(
        default=False,
        description="Enable orchestrator-level in-flight awareness",
    )

    max_total_in_flight: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Max rows in flight across all batching plugins",
    )
```

---

## 11. Migration Plan

### Phase 1: Core Infrastructure (Week 1)

| Task | Description | Risk | Owner |
|------|-------------|------|-------|
| Implement `RowReorderBuffer` | Core FIFO buffer with notify() fix | Low | Backend |
| Implement `BatchTransformMixin` | Integration mixin with traceback capture | Low | Backend |
| Implement `BatchingPluginProtocol` | Standardized interface | Low | Backend |
| Implement `BatchMetrics` | Metrics dataclass | Low | Backend |
| Unit tests (P0) | All tests from Section 9.2 | Medium | QA |

### Phase 2: LLM Transform Integration (Week 2)

| Task | Description | Risk | Owner |
|------|-------------|------|-------|
| Update `AzureMultiQueryLLMTransform` | Add mixin, integrate | Medium | Backend |
| Update `OpenRouterMultiQueryTransform` | Add mixin, integrate | Medium | Backend |
| Integration tests | E2E with Orchestrator + Landscape | Medium | QA |
| Stress tests | 1000-row load tests | Medium | QA |

### Phase 3: Coordination & Documentation (Week 3)

| Task | Description | Risk | Owner |
|------|-------------|------|-------|
| Implement `GlobalInFlightLimiter` | Optional cross-plugin coordination | Low | Backend |
| Performance benchmarks | Validate 3x throughput claim | Low | QA |
| Update plugin authoring guide | Document BatchTransformMixin | Low | Docs |
| Update CLAUDE.md | Add batching patterns | Low | Docs |

---

## 12. Success Criteria

| Metric | Target | Measurement | Status |
|--------|--------|-------------|--------|
| FIFO invariant | 100% | Property tests, stress tests | Pending |
| Audit determinism | 100% | E2E test verifies node_states order | Pending |
| Throughput improvement | ≥2.5x | Benchmark with LLM transforms | Pending |
| No deadlocks | 0 occurrences | Shutdown/error tests | Pending |
| No orchestrator changes | 0 lines | Code review | ✅ Verified |
| Backward compatible | 100% | Existing transforms work unchanged | Pending |
| Protocol compliance | 100% | All batching plugins implement protocol | Pending |

---

## 13. Appendix A: Why Not Orchestrator-Level Pipelining?

An earlier design considered orchestrator-level pipelining with a `ReleaseQueue`. That approach was rejected because:

| Concern | Orchestrator Pipelining | Plugin Pipelining |
|---------|------------------------|-------------------|
| **Orchestrator changes** | ~1,000 lines new code | **0 lines** |
| **ADR-001** | Superseded | **Extended** |
| **Audit determinism** | Requires careful design | **Preserved automatically** |
| **Fork/join** | Complex state machine | **Unchanged** |
| **Recovery** | New checkpoint complexity | **Unchanged** |
| **Failure scope** | Global | **Per-plugin** |
| **Systemic risk** | Convoy effect, deadlocks | **Contained** |

**The key insight:** By pushing pipelining to the plugin boundary, we get the throughput benefits without the systemic risks. The orchestrator remains a simple, deterministic loop.

---

## 14. Appendix B: Review Board Feedback Integration

### Architecture Review Findings

| Finding | Resolution | Section |
|---------|------------|---------|
| Remove defensive ctx.token fallback | Fixed in mixin | 4.2 |
| Document thread pool sizing constraint | Added explicit guidance | 4.1 |
| Add empirical benchmarks | Added to test strategy | 9.3 |

### Python Engineering Review Findings

| Finding | Resolution | Section |
|---------|------------|---------|
| `notify_all()` → `notify()` | Fixed in buffer | 3.1 |
| Add type hints for `_batch_futures` | Added to mixin | 4.1 |
| Capture traceback in exception handling | Fixed in mixin | 4.1 |
| Remove timeout polling in submit() | Fixed in buffer | 3.1 |

### Quality Engineering Review Findings

| Finding | Resolution | Section |
|---------|------------|---------|
| Missing 1000-row stress test | Added | 9.2.3 |
| Missing shutdown deadlock test | Added | 9.2.2 |
| Missing E2E with Landscape | Added | 9.2.5 |
| Missing audit ordering test | Added | 9.2.5 |
| Missing double-complete safety test | Added | 9.2.2 |

### Systems Thinking Review Findings

| Finding | Resolution | Section |
|---------|------------|---------|
| Cross-plugin coordination | Added GlobalInFlightLimiter | 7 |
| Standardize batch behavior contract | Added BatchingPluginProtocol | 5 |
| Document crash recovery | Added explicit section | 8 |
| Limit initial adoption scope | Added phased rollout | 6.2 |

---

## 15. References

- **ADR-001:** `docs/design/adr/001-plugin-level-concurrency.md` (extended, not superseded)
- **ReorderBuffer:** `src/elspeth/plugins/pooling/reorder_buffer.py` (pattern source)
- **PooledExecutor:** `src/elspeth/plugins/pooling/executor.py` (shared pool)
- **Review Board Report:** 2026-01-26 (4-perspective review)
