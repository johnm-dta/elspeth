# ADR-001: Plugin-Level Concurrency (Push Complexity to the Edges)

**Date:** 2026-01-22
**Status:** Accepted
**Deciders:** Architecture team, Core maintainers
**Tags:** concurrency, architecture, orchestrator, plugins, audit-integrity

## Context

ELSPETH pipelines process rows through a DAG of transforms (sources → transforms → sinks), and many operations involve I/O-bound external calls:

- **LLM API calls:** Batches of 50-1000 rows calling GPT-4, Claude, etc.
- **Database writes:** Bulk inserts to sink databases
- **HTTP API calls:** External validation services, webhooks
- **Azure Blob Storage:** Reading/writing large files

**The Question:** Where should concurrency live?

1. **Orchestrator-level concurrency:** Process multiple rows in parallel through the entire pipeline
2. **Plugin-level concurrency:** Plugins internally parallelize their work (e.g., LLM calls, DB writes)

**Critical Constraint:** ELSPETH is an **audit-first system**. Every row's journey through the DAG must be traceable with perfect fidelity. The `landscape` audit database records:

- Every row loaded from source
- Every transform input/output state
- Every routing decision
- Every external call (request hash, response hash, latency)
- Every token's terminal outcome

**Auditability requirements (CLAUDE.md):**

> "I don't know what happened" is never an acceptable answer for any output

> Every token reaches exactly one terminal state - no silent drops

This means:
- Deterministic execution order for audit reproducibility
- Clear sequencing of node_states (token_id, node_id, attempt)
- No race conditions in audit trail recording

## Decision

**Concurrency lives at the plugin boundary, NOT at the orchestrator level.**

The orchestrator processes rows **sequentially** through the DAG. Plugins MAY internally parallelize their operations using thread pools or async I/O.

### Key Principles

1. **Orchestrator is single-threaded and deterministic**
   - Processes one token at a time through the DAG
   - Records audit events in strict sequential order
   - No locking, no race conditions, no ordering ambiguity

2. **Plugins own their concurrency strategy**
   - `PooledExecutor` for parallel LLM calls (thread pool, reorder buffer)
   - Database sinks can batch-write rows internally
   - Azure blob operations can use async I/O

3. **Audit boundaries are plugin method calls**
   - Orchestrator calls `transform.process(row, ctx)` sequentially
   - Inside `process()`, the plugin can parallelize as needed
   - Orchestrator records the result in `node_states` table

4. **Configuration declares intent, not mechanism**
   - `ConcurrencySettings.max_workers` exists for future plugin use
   - NOT used by orchestrator for row-level parallelism
   - Plugins read this config if they want orchestrator-wide concurrency limits

### What This Looks Like in Practice

**Orchestrator loop (single-threaded):**

```python
for token in work_queue:
    # Sequential processing - one token at a time
    result = transform.process(token.row_data, ctx)  # Plugin boundary
    recorder.record_node_state(...)  # Audit
    work_queue.push_downstream(result)
```

**Inside LLM transform plugin (parallel):**

```python
class AzureBatchLLMTransform:
    def __init__(self, config):
        # Plugin creates its own thread pool
        self.pool = PooledExecutor(max_workers=config.concurrency.max_workers)

    def process_batch(self, rows: list[dict]) -> list[TransformResult]:
        # Parallelize the 50 LLM calls using thread pool
        futures = [self.pool.submit(self._call_llm, row) for row in rows]
        results = [f.result() for f in futures]  # Blocks until all complete
        return results  # Orchestrator records these sequentially
```

**Inside database sink (bulk writes):**

```python
class DatabaseSink:
    def write(self, rows: list[dict]) -> ArtifactDescriptor:
        # Internally batch the INSERT statements
        with self.engine.begin() as conn:
            conn.execute(self.table.insert(), rows)  # Single bulk write
        return ArtifactDescriptor(...)
```

## Consequences

### Positive Consequences

1. **Audit trail integrity is guaranteed**
   - No race conditions in `node_states` recording
   - Deterministic order: `(token_id, node_id, attempt)` uniqueness constraint works
   - Replay reproducibility: re-running produces identical audit records

2. **Orchestrator remains simple**
   - No thread pool management
   - No locking, no mutexes, no concurrent data structures
   - Easy to reason about: "one token, one path, sequential steps"
   - Checkpoint/recovery logic is straightforward

3. **Plugins optimize where it matters**
   - LLM plugins parallelize API calls (I/O-bound)
   - Database sinks batch writes (I/O-bound)
   - CPU-bound transforms still run sequentially (rare in ELSPETH workflows)

4. **Clear performance boundaries**
   - Slow pipeline? Profile plugin internals, not orchestrator
   - LLM latency? Check `PooledExecutor` tuning
   - Database bottleneck? Check sink batch size

5. **Aligns with "push complexity to edges" principle**
   - Core is simple and correct
   - Edges (plugins) are sophisticated and optimized
   - Classic systems architecture pattern

### Negative Consequences

1. **CPU-bound transforms cannot parallelize across rows**
   - If you have a transform that does heavy computation (uncommon), it processes rows sequentially
   - Mitigation: Rare in ETL/LLM pipelines (most work is I/O-bound)

2. **Cannot pipeline across stages**
   - Source → Transform → Sink is sequential (can't have source producing while transform processes)
   - Mitigation: Plugins can internally buffer (e.g., aggregations, batch processing)

3. **Configuration can be misleading**
   - `ConcurrencySettings.max_workers` exists but orchestrator doesn't use it
   - Requirement PRD-005 marked as DIVERGED
   - Mitigation: Documentation clarifies this is for plugin use

### Neutral Consequences

- Plugins must implement their own thread safety if they parallelize
- Performance tuning is plugin-specific (no one-size-fits-all knob)
- Each plugin can choose its own concurrency strategy (threads, async, processes)

## Alternatives Considered

### Alternative 1: Orchestrator-Level Row Parallelism

**Description:**

Use a thread pool or process pool in the orchestrator to process multiple tokens concurrently through the DAG:

```python
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(process_token, token) for token in batch]
    wait(futures)
```

**Rejected because:**

1. **Audit trail ordering becomes non-deterministic**
   - Multiple threads writing to `node_states` table concurrently
   - Need locking around SQLAlchemy calls (performance hit)
   - `(token_id, node_id, attempt)` records can arrive out of order
   - Replay produces different audit record sequences

2. **Checkpoint/recovery becomes complex**
   - Need to track in-flight tokens (what's in the thread pool?)
   - Crash recovery: which tokens were processing when we crashed?
   - Resume logic: need to coordinate thread pool state

3. **DAG execution ordering ambiguity**
   - If token A and token B both go through transform T, which runs first?
   - Matters for aggregations (batch order affects batch composition)
   - Matters for stateful transforms (though we discourage these)

4. **Plugin concurrency becomes nested**
   - Orchestrator parallelizes rows
   - Plugin parallelizes work within its method
   - Result: oversubscription (N orchestrator threads × M plugin threads)

### Alternative 2: Async/Await Throughout

**Description:**

Use Python's `async`/`await` for the entire pipeline stack:

```python
async def orchestrator_loop():
    async for token in work_queue:
        result = await transform.process_async(token, ctx)
        await recorder.record_node_state_async(...)
```

**Rejected because:**

1. **Viral async transformation required**
   - Every plugin must be `async` (including third-party library integrations)
   - SQLAlchemy Core (our database layer) has limited async support
   - Many libraries we depend on are sync-only (pandas, rfc8785, etc.)

2. **Concurrency still needs careful management**
   - Just because it's async doesn't mean we can parallelize audit writes
   - Still need sequential node_state recording for audit integrity
   - Async is about I/O concurrency, not parallel execution ordering

3. **Complexity without clear benefit**
   - Plugins can already use async internally (e.g., `aiohttp` in HTTP transforms)
   - Orchestrator being sync doesn't prevent plugin async

### Alternative 3: Actor Model (Message Passing)

**Description:**

Use an actor system (e.g., Ray, Dramatiq) where each node in the DAG is an actor processing messages:

```python
@ray.remote
class TransformActor:
    def process(self, token):
        result = self.transform.process(token.data)
        recorder_actor.record.remote(result)
```

**Rejected because:**

1. **Massive architectural complexity**
   - Distributed systems concerns (failure handling, message ordering)
   - External dependencies (Ray runtime, Redis for message queue)
   - Deployment complexity (need to run Ray cluster)

2. **Audit trail becomes eventual consistency problem**
   - Actors record audit events asynchronously
   - Need distributed transaction coordination to ensure consistency
   - Crash recovery: need to rebuild orchestrator state from message logs

3. **Overkill for target use cases**
   - ELSPETH pipelines are typically 10K-100K rows, not billions
   - I/O-bound workloads (LLM APIs) dominate CPU-bound workloads
   - Plugin-level concurrency is sufficient

## Related Decisions

- **Requirements PRD-005:** Marked as DIVERGED in `docs/architecture/requirements.md`
- **LND-011:** `node_states` table design assumes sequential recording
- **AUD-001:** "Every token reaches exactly one terminal state" - requires deterministic orchestrator

## References

- `src/elspeth/engine/orchestrator.py:88-816` - Sequential token processing loop
- `src/elspeth/plugins/pooling/executor.py:90` - Plugin-level thread pool (`PooledExecutor`)
- `src/elspeth/core/config.py:469-473` - `ConcurrencySettings.max_workers` (for plugin use)
- `docs/architecture/requirements.md` - PRD-005 marked as DIVERGED
- [Wikipedia: End-to-end principle](https://en.wikipedia.org/wiki/End-to-end_principle) - "Push complexity to edges"

## Notes

### Future Considerations

1. **Streaming/Incremental Processing**
   - If we add streaming sources (Kafka, websockets), we may need async I/O for backpressure
   - Can be handled by making the source plugin async internally
   - Orchestrator loop can still be synchronous (poll source, process token, repeat)

2. **Multi-Core CPU-Bound Transforms**
   - If a use case emerges with heavy CPU transforms, consider:
     - Batch processing within plugin (process N rows in parallel, return N results)
     - External compute (transform calls Ray/Spark/Dask for parallel computation)
   - Don't change orchestrator concurrency model

3. **Distributed ELSPETH**
   - If we need to scale beyond single-node capacity:
     - Shard runs across multiple orchestrator instances (run-level parallelism)
     - Each orchestrator still processes its assigned run sequentially
     - Don't introduce row-level parallelism within orchestrator

### Implementation Status

- ✅ Orchestrator is single-threaded (verified)
- ✅ `PooledExecutor` in LLM plugins uses thread pool (verified)
- ✅ Database sinks use bulk writes (verified)
- ✅ Audit trail integrity maintained (verified via tests)
- ❌ Documentation update needed: Add section to architecture.md explaining this decision

---

**Revision History:**

- 2026-01-22: Initial ADR (accepted)
