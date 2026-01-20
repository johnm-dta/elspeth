# Pooled LLM Queries Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create the implementation plan from this design.

**Goal:** Enable parallel LLM API calls within a single transform while maintaining strict row order and gracefully handling capacity errors.

**Architecture:** Per-transform pooled executor with semaphore-controlled dispatch, AIMD throttle for adaptive rate control, and reorder buffer for strict output ordering.

**Tech Stack:** `concurrent.futures.ThreadPoolExecutor`, existing `AuditedHttpClient`, existing retry infrastructure.

---

## Problem Statement

Current LLM transforms process rows sequentially (~1-1.5s per API call). For 100k row runs, this means ~28+ hours of wall-clock time. The target API server is "flaky" - it varies between allowing 20 req/s and 5 req/s, returning capacity errors (429, 503, 529) unpredictably.

**Requirements:**
1. Parallel API calls to maximize throughput
2. Strict output ordering (row N always emits before row N+1)
3. Capacity errors never fail rows - infinite retry until success
4. Adaptive throttling to avoid hammering overloaded servers
5. Full audit trail of parallel execution

---

## Configuration Schema

Pool configuration lives in each LLM transform's options:

```yaml
row_plugins:
  - plugin: openrouter_llm
    options:
      model: "anthropic/claude-3-haiku"
      template: "..."

      # Parallel execution settings (all optional with defaults)
      pool_size: 10                   # Max concurrent requests (default: 1 = sequential)

      # AIMD throttle settings
      min_dispatch_delay_ms: 0        # Floor for delay between dispatches (default: 0)
      max_dispatch_delay_ms: 5000     # Ceiling for delay (default: 5000)
      backoff_multiplier: 2.0         # Multiply delay on capacity error (default: 2.0)
      recovery_step_ms: 50            # Subtract from delay on success (default: 50)
```

**Backwards Compatibility:**
- `pool_size: 1` (default) = sequential processing, no buffering, no throttle logic
- `pool_size > 1` = pooled mode with reorder buffer and AIMD throttle

**Capacity Error Codes (universal):**
- 429 (Too Many Requests)
- 503 (Service Unavailable)
- 529 (Overloaded - Azure, some other providers)

---

## Execution Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PooledLLMExecutor                              │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │ Incoming │───►│  Dispatcher  │───►│   In-Flight Pool        │   │
│  │   Rows   │    │ (semaphore)  │    │   (max: pool_size)      │   │
│  └──────────┘    └──────────────┘    └───────────┬─────────────┘   │
│                         │                        │                  │
│                         │ throttle               │ completions      │
│                         │ delay                  ▼                  │
│                  ┌──────┴───────┐    ┌─────────────────────────┐   │
│                  │    AIMD      │◄───│    Reorder Buffer       │   │
│                  │  Throttle    │    │ (emit in submit order)  │   │
│                  └──────────────┘    └───────────┬─────────────┘   │
│                                                  │                  │
│                                                  ▼                  │
│                                      ┌─────────────────────────┐   │
│                                      │   Ordered Results       │   │
│                                      └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Rows enter the dispatcher, which respects the semaphore (max `pool_size` in flight)
2. Dispatcher waits `current_delay` between dispatches (AIMD-controlled)
3. Requests complete (possibly out of order) and enter the reorder buffer
4. Buffer emits results strictly in submission order
5. Capacity errors feed back to AIMD throttle (increase delay); successes decrease delay

The executor is **synchronous from the caller's perspective** - the transform's `process()` method returns results in order, hiding the parallelism internally.

---

## Error Classification & Retry Behavior

### Capacity Errors (never fail the row)

**Triggers:**
- HTTP 429 (Too Many Requests)
- HTTP 503 (Service Unavailable)
- HTTP 529 (Overloaded - Azure and others)
- Connection timeout during high load

**Behavior:**
- Trigger AIMD throttle (multiply `current_delay` by `backoff_multiplier`)
- Re-queue the row for retry (stays in pool, doesn't count against `max_attempts`)
- Row remains in-flight until success or transform-level timeout (optional, e.g., 30 minutes)

### Normal Errors (use existing retry config)

**Triggers:**
- HTTP 400 (Bad Request - malformed prompt)
- HTTP 401/403 (Auth errors)
- HTTP 500 (Server error)
- Malformed JSON response
- Template rendering errors

**Behavior:**
- Use existing `retry` config (`max_attempts`, exponential backoff)
- After exhausting retries → row fails with `TransformResult.error()`
- Recorded in audit trail with full error details

### AIMD Throttle Algorithm

```python
current_delay = 0  # Start with no delay

# On capacity error:
current_delay = min(current_delay * backoff_multiplier, max_dispatch_delay_ms)
# If current_delay was 0, set to a minimum (e.g., 100ms) to start backing off

# On success:
current_delay = max(current_delay - recovery_step_ms, min_dispatch_delay_ms)
```

**Key property:** Fast ramp-down (multiplicative), slow ramp-up (additive). This prevents "riding the edge" where you're constantly hitting capacity limits.

The throttle state is **shared across all in-flight requests** for that transform instance, so one capacity error slows down all subsequent dispatches.

---

## Audit Trail & Observability

### Calls Table (existing structure)

Each HTTP request gets its own `call_id`, even for parallel requests:
- `state_id` → links to the node_state for this transform execution
- `call_index` → distinguishes multiple calls for same row (retries)
- `latency_ms` → actual request duration
- `status` → success/failure/capacity_retry

### New Fields in Node State Context

```json
{
  "context_after_json": {
    "pool_config": {
      "pool_size": 10,
      "dispatch_delay_at_completion_ms": 150
    },
    "pool_stats": {
      "capacity_retries": 3,
      "max_concurrent_reached": 10,
      "total_throttle_time_ms": 2340
    }
  }
}
```

### Ordering Metadata

- `submit_index` → order row was submitted to pool
- `complete_index` → order row's request completed

These let auditors verify reordering worked correctly and identify any "lost" rows if a batch fails mid-execution.

### Run-Level Summary

Stored in runs table `settings_json`:
- Total capacity retries across all rows
- Peak throttle delay reached
- Time spent throttled vs executing

---

## Implementation Architecture

### New Files

```
src/elspeth/plugins/llm/
├── base.py                    # Existing - unchanged
├── openrouter.py              # Minimal changes - delegates to pooled executor
├── azure.py                   # Minimal changes - delegates to pooled executor
├── pooled_executor.py         # NEW - PooledExecutor class
└── aimd_throttle.py           # NEW - AIMD throttle state machine
```

### PooledExecutor Responsibilities

- Manages the semaphore (pool_size)
- Maintains reorder buffer
- Coordinates with AIMD throttle
- Uses `concurrent.futures.ThreadPoolExecutor` for parallel HTTP calls
- Returns results in submission order

### LLM Transform Changes (minimal)

```python
# In OpenRouterLLMTransform.__init__:
if cfg.pool_size > 1:
    self._executor = PooledExecutor(
        pool_size=cfg.pool_size,
        throttle_config=cfg.throttle_config,
        http_client=self._http_client,
    )
else:
    self._executor = None  # Sequential mode

# In process():
if self._executor:
    return self._executor.execute(row, ctx, self._build_request)
else:
    return self._execute_single(row, ctx)  # Existing logic
```

### Capacity Error Detection

In `src/elspeth/plugins/clients/http.py`:

```python
CAPACITY_ERROR_CODES = frozenset({429, 503, 529})
```

The existing `AuditedHttpClient` already records full request/response - it just needs to surface whether the error was capacity-related.

---

## Testing Strategy

### Unit Tests (fast, mocked HTTP)

1. **Reorder buffer correctness:**
   - Submit rows [0,1,2,3,4], complete in order [2,0,4,1,3]
   - Verify output order is [0,1,2,3,4]

2. **AIMD throttle behavior:**
   - Verify delay multiplies on capacity error
   - Verify delay subtracts on success
   - Verify bounds (min/max) are respected
   - Verify asymmetry (fast down, slow up)

3. **Capacity error classification:**
   - 429 → capacity retry, no failure count
   - 503 → capacity retry
   - 529 → capacity retry
   - 500 → normal retry with count
   - 400 → normal retry with count

4. **Pool size edge cases:**
   - `pool_size: 1` → sequential behavior unchanged
   - `pool_size: 5`, only 3 rows → no deadlock, all complete
   - End-of-source with in-flight requests → drain cleanly

### Integration Tests (real threading)

5. **Order preservation under load:**
   - 100 rows, pool_size 10, random latencies (50-500ms)
   - Verify output order matches input order

6. **Throttle under simulated capacity errors:**
   - Mock server returns 429 for 20% of requests
   - Verify all rows eventually succeed
   - Verify throttle delay increased then recovered

### Property-Based Tests (Hypothesis)

7. **Reorder buffer invariant:**
   - For any completion order permutation, output order equals submission order

---

## Open Questions (resolved)

1. ~~Where should parallelism happen?~~ → At the LLM transform level
2. ~~How should concurrency be configured?~~ → Per-transform `pool_size`
3. ~~How should rows be accumulated?~~ → Eager dispatch with semaphore
4. ~~How should capacity errors be handled?~~ → Infinite retry with AIMD throttle
5. ~~What about order preservation?~~ → Strict original order via reorder buffer

---

## Summary

| Aspect | Decision |
|--------|----------|
| Parallelism scope | Per-transform (not row-level) |
| Configuration | Per-transform `pool_size` with AIMD settings |
| Dispatch model | Eager with semaphore |
| Capacity handling | Infinite retry, never fails row |
| Throttle algorithm | AIMD (fast down, slow up) |
| Order guarantee | Strict submission order |
| Backwards compat | `pool_size: 1` = unchanged behavior |
