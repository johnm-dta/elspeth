# Unify Transform API to `process()` - Tasking Statement

**Date:** 2026-01-26
**Status:** Approved for implementation
**Architect Review:** Architecture Critic assessment completed (Score: 2/5 for current dual-API state)

## Problem Statement

The current codebase has two incompatible transform APIs:

1. **`process(row, ctx) -> TransformResult`** - Synchronous, used by simple transforms
2. **`accept(row, ctx) -> None`** + `flush_batch_processing()` - Asynchronous push-based, used by `BatchTransformMixin`

This creates three critical issues:

### 1. Liskov Substitution Principle Violation (Critical)

Transforms using `BatchTransformMixin` implement `TransformProtocol` but raise `NotImplementedError` in `process()`:

```python
# src/elspeth/plugins/llm/azure.py:225-242
def process(self, row: dict, ctx: PluginContext) -> TransformResult:
    raise NotImplementedError(
        "AzureLLMTransform uses row-level pipelining. Use accept() instead."
    )
```

Code operating on `TransformProtocol` cannot safely call `process()` without type-checking.

### 2. Engine Incompatibility (High)

The engine's `TransformExecutor.execute_transform()` calls `process()` unconditionally:

```python
# src/elspeth/engine/executors.py:192
result = transform.process(row, ctx)
```

There is NO dispatch logic for `accept()`. A batch transform wired into a standard pipeline crashes immediately.

### 3. Audit Trail Bypass (High)

Rows processed via `accept()` → `BatchTransformMixin._process_row()` bypass `TransformExecutor` entirely. This means:
- No `node_states` entries recorded
- No external call recording (no `ctx.state_id` set)
- Violates ELSPETH's core auditability requirement

## Solution: Unify to `process()`

Make `process()` the universal interface. Batch transforms implement it by submitting to internal batch infrastructure and blocking until the result is ready.

### Target Architecture

```python
class AzureLLMTransform(BaseTransform, BatchTransformMixin):
    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Engine calls this. Submit to batch, wait for result, return it."""
        # Submit row to batch infrastructure, get a future
        future = self._submit_row_to_batch(row, ctx)

        # Block until THIS ROW completes (other rows process concurrently)
        return future.result(timeout=self._processing_timeout)
```

### Key Insight

The engine already calls `process()` in FIFO order. The concurrency happens INSIDE the batch infrastructure - multiple `process()` calls can have their work executing concurrently in the thread pool. From the engine's perspective, each `process()` call is just "slow" - but internally, rows are processed in parallel.

```
Engine Thread:              Batch Worker Threads:
─────────────              ────────────────────
process(row1) ─────────┬──► [Worker 1: LLM call for row1]
  │ (blocks)           │
  │                    ├──► [Worker 2: LLM call for row2]  ← started by process(row2)
  │                    │
  ▼ (row1 done)        │
return result1         │
                       │
process(row2) ─────────┤
  │ (already running!) │
  ▼ (row2 done)        │
return result2         ▼
```

### What Changes

| Component | Current | After |
|-----------|---------|-------|
| `BatchTransformMixin.accept()` | Public API | Internal `_submit_row()` |
| `BatchTransformMixin.flush_batch_processing()` | Called by tests/caller | Called in `close()` |
| `AzureLLMTransform.process()` | Raises `NotImplementedError` | Submits to batch, blocks, returns result |
| `connect_output()` | Called by caller before `accept()` | Called in `on_start()` by engine/orchestrator |
| Engine `TransformExecutor` | Unchanged | Unchanged |
| Audit trail | Bypassed for batch transforms | Works automatically |

### What Stays the Same

- `BatchTransformMixin` internal architecture (thread pool, semaphore, FIFO reorder buffer)
- `CollectorOutputPort` (still useful for testing)
- `TokenInfo` tracking through the pipeline
- All existing simple transforms

## Implementation Tasks

### Phase 1: Modify BatchTransformMixin

1. **Add `_submit_row()` method** that returns a `Future[TransformResult]`
   - Submits row to thread pool
   - Returns immediately with a future
   - The future resolves when that specific row completes

2. **Add result tracking per-row**
   - Currently `_release_loop` emits to output port
   - Need to also resolve the future for that row
   - Use `row_id` or sequence number to correlate

3. **Rename `accept()` to `_accept_row_internal()`** (internal use only)

4. **Move `flush_batch_processing()` to `close()`**
   - Drain all pending work on shutdown
   - Respect timeout

### Phase 2: Modify LLM Transforms

For each transform using `BatchTransformMixin`:
- `AzureLLMTransform` (`src/elspeth/plugins/llm/azure.py`)
- `OpenRouterLLMTransform` (`src/elspeth/plugins/llm/openrouter.py`)
- `AzureMultiQueryLLMTransform` (`src/elspeth/plugins/llm/azure_multi_query.py`)
- `OpenRouterMultiQueryLLMTransform` (`src/elspeth/plugins/llm/openrouter_multi_query.py`)

1. **Implement `process()` properly:**
   ```python
   def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
       future = self._submit_row(row, ctx)
       return future.result(timeout=60.0)
   ```

2. **Call `connect_output()` in `on_start()`**
   - Engine provides context with output port reference
   - Or: create internal output port in `on_start()`

3. **Remove `accept()` from public interface**

### Phase 3: Update Tests

1. **Revert test changes** that converted `process()` to `accept()`
   - `test_azure.py`
   - `test_openrouter.py`
   - `test_azure_multi_query.py`
   - `test_openrouter_multi_query.py`
   - `test_azure_multi_query_retry.py`
   - `test_azure_multi_query_profiling.py`

2. **Tests should use `process()` like any other transform:**
   ```python
   result = transform.process(row, ctx)
   assert result.status == "success"
   ```

3. **Keep `CollectorOutputPort` tests** for internal BatchTransformMixin testing

### Phase 4: Verify Audit Trail

1. Confirm `TransformExecutor.execute_transform()` works with batch transforms
2. Verify `node_states` entries are recorded
3. Verify external calls are recorded via `ctx.state_id`

## Risk Mitigation

### Deadlock Risk

**Concern:** Blocking on `future.result()` while batch workers are running could deadlock.

**Mitigation:**
- Thread pool is separate from the calling thread
- Semaphore controls concurrency within the pool, not blocking on results
- Add timeout to `future.result()` call
- Test with high concurrency (100+ rows)

### Performance Risk

**Concern:** Blocking on each row's result may reduce throughput.

**Mitigation:**
- The batch infrastructure still processes rows concurrently
- Row N+1's `process()` call can submit to batch immediately
- The "blocking" is just waiting for completion, not preventing new submissions
- Benchmark before/after to verify

### Test Execution Order

**Concern:** Tests may have become dependent on `accept()`/`flush()` timing.

**Mitigation:**
- `process()` is synchronous from caller's perspective
- Tests become simpler (no need to flush)
- FIFO ordering is guaranteed by engine call order

## Verification Criteria

1. All transforms implement `process()` without raising `NotImplementedError`
2. `TransformExecutor.execute_transform()` works with batch transforms
3. Audit trail records all batch transform operations
4. All existing tests pass (or are updated to use `process()`)
5. No deadlocks under concurrent load
6. Performance is acceptable (benchmark TBD)

## Files to Modify

### Core Infrastructure
- `src/elspeth/plugins/batching/mixin.py` - Add future-based result tracking

### LLM Transforms
- `src/elspeth/plugins/llm/azure.py`
- `src/elspeth/plugins/llm/openrouter.py`
- `src/elspeth/plugins/llm/azure_multi_query.py`
- `src/elspeth/plugins/llm/openrouter_multi_query.py`

### Tests (revert to process() API)
- `tests/plugins/llm/test_azure.py`
- `tests/plugins/llm/test_openrouter.py`
- `tests/plugins/llm/test_azure_multi_query.py`
- `tests/plugins/llm/test_openrouter_multi_query.py`
- `tests/plugins/llm/test_azure_multi_query_retry.py`
- `tests/plugins/llm/test_azure_multi_query_profiling.py`
- `tests/plugins/batching/test_batch_transform_mixin.py`

## Open Questions

1. **Output port wiring:** How does the engine provide the output port to batch transforms? Via `ctx`? Via `on_start()`? Or do batch transforms create their own internal port?

2. **Multi-row results:** `AzureMultiQueryLLMTransform` processes 4 queries per row. Should `process()` return when all 4 complete? (Yes - this is row-level atomicity)

3. **Timeout configuration:** Where should the `future.result(timeout=...)` value come from? Config? Fixed? Per-transform?

## References

- Architecture Critic Assessment: Session 2026-01-26, Agent a2677e5
- `BatchTransformMixin` implementation: `src/elspeth/plugins/batching/mixin.py`
- `TransformExecutor`: `src/elspeth/engine/executors.py`
- Row-level pipelining design: `docs/design/row-level-pipelining-design.md`
