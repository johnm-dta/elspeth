# Azure Multi-Query Plugin - Critical Implementation Issues

**Date:** 2026-01-26
**Severity:** ❌ **CRITICAL - Current implementation does NOT match expected behavior**

---

## Executive Summary

The Azure Multi-Query plugin profiling revealed **two critical implementation issues** that contradict the expected production behavior:

1. ❌ **NO capacity error retries** - Plugin fails immediately instead of retrying with AIMD backoff
2. ❌ **Sequential row processing** - Wastes pool capacity by processing one row at a time

**Impact:**
- Rows fail unnecessarily when rate limits are hit (should retry)
- Poor parallelism utilization (pool_size=100 only uses ~10 workers at a time)
- Previous profiling report measured WRONG behavior

---

## Issue 1: No Capacity Error Retries ❌

### Current Behavior (WRONG)

When a query hits a rate limit (`RateLimitError`), the plugin:
1. Converts it to `CapacityError` (line 226)
2. Catches it and immediately fails (line 439-447)
3. Marks the entire row as failed

```python
# azure_multi_query.py line 226
except RateLimitError as e:
    raise CapacityError(429, str(e)) from e

# Line 439-447 (_execute_queries_parallel)
except CapacityError as e:
    # If capacity error escapes, treat as error
    results_by_index[idx] = TransformResult.error({
        "reason": "capacity_exhausted",
        "query": self._query_specs[idx].output_prefix,
        "error": str(e),
    })
```

**Result:** Row fails permanently, no retry attempted.

### Expected Behavior (CORRECT)

Capacity errors should trigger **automatic retries with AIMD backoff** until `max_capacity_retry_seconds` is exceeded.

The `PooledExecutor` class already implements this correctly (lines 257-310 in `executor.py`):

```python
while True:  # RETRY LOOP
    try:
        result = process_fn(row, state_id)
        self._throttle.on_success()
        return result  # Success
    except CapacityError as e:
        # Check max retry time
        if time.monotonic() >= max_time:
            return TransformResult.error({
                "reason": "capacity_retry_timeout",
                ...
            })

        # AIMD backoff
        self._throttle.on_capacity_error()  # Increase delay

        # Release semaphore (let others work)
        self._semaphore.release()

        # Wait with exponentially increasing delay
        time.sleep(retry_delay_ms / 1000)

        # Re-acquire and RETRY
        self._semaphore.acquire()
        # Continue loop
```

**Retry behavior:**
- Initial delay: ~100ms
- Each failure: delay doubles (AIMD backoff)
- Max delay: configurable (e.g., 30s)
- Timeout: configurable (e.g., 300s total retry time)

**Result:** Row succeeds after transient rate limit passes, OR fails after retry timeout.

### Why This Happened

The plugin uses `ThreadPoolExecutor` directly (line 424) instead of `PooledExecutor.execute_batch()`:

```python
# CURRENT (WRONG)
with ThreadPoolExecutor(max_workers=self._executor.pool_size) as executor:
    futures = {executor.submit(self._process_single_query, ...): i ...}
    # No retry logic!
```

**Should be:**

```python
# EXPECTED (CORRECT)
contexts = [RowContext(row=row, state_id=state_id, row_index=i) for i in ...]
results = self._executor.execute_batch(contexts, self._process_single_query)
# PooledExecutor handles retries internally
```

The comment at line 402-408 claims:

> "All queries share the same underlying AzureOpenAI client which handles its own rate limiting, so AIMD overhead is unnecessary."

**This is incorrect.** The `AuditedLLMClient` does NOT retry - it raises `RateLimitError` immediately (line 203-218 in `llm.py`).

---

## Issue 2: Sequential Row Processing ❌

### Current Behavior (WRONG)

Batch processing is sequential (line 519 in `_process_batch`):

```python
for row in rows:  # ONE AT A TIME
    result = self._process_single_row_internal(row, ctx.state_id)
    ...
```

**Example:** pool_size=100, 10 queries per row, 50 rows

| Row | Workers Used | Workers Idle | Efficiency |
|-----|--------------|--------------|------------|
| Row 1 | 10 | 90 | 10% |
| Row 2 | 10 | 90 | 10% |
| Row 3 | 10 | 90 | 10% |
| ... | ... | ... | ... |
| Row 50 | 10 | 90 | 10% |

**Result:** 90% of the pool sits idle! ❌

### Expected Behavior (CORRECT)

With pool_size=100 and 10 queries per row, process **10 rows simultaneously**:

| Batch | Workers Used | Workers Idle | Efficiency |
|-------|--------------|--------------|------------|
| Rows 1-10 | 100 (10 per row) | 0 | 100% |
| Rows 11-20 | 100 (10 per row) | 0 | 100% |
| Rows 21-30 | 100 (10 per row) | 0 | 100% |
| ... | ... | ... | ... |
| Rows 41-50 | 100 (10 per row) | 0 | 100% |

**Result:** Full pool utilization, 10x faster batch processing ✅

### How to Fix

Use `PooledExecutor.execute_batch()` with **cross-row parallelism**:

```python
# Flatten: (rows × queries) → single list of work items
contexts = []
for row_idx, row in enumerate(rows):
    for query_spec in self._query_specs:
        contexts.append(RowContext(
            row={"row_data": row, "spec": query_spec},
            state_id=f"{ctx.state_id}_r{row_idx}",
            row_index=row_idx * len(self._query_specs) + query_spec_idx,
        ))

# Execute all queries for all rows in parallel
all_results = self._executor.execute_batch(contexts, self._process_single_query)

# Group results back by row (4 queries per row)
for row_idx in range(len(rows)):
    start = row_idx * len(self._query_specs)
    end = start + len(self._query_specs)
    row_results = all_results[start:end]
    # Check atomicity and merge
```

This fully utilizes the pool (100 workers processing 100 queries simultaneously).

---

## Atomicity Guarantees with Concurrent Row Processing

### Question from User

"If we have 2×5 queries (10 total per row) and pool_size=50 or 100, can we process 5-10 rows simultaneously and still maintain atomicity guarantees?"

### Answer: YES ✅ (But requires careful implementation)

**Key insight:** Atomicity is a **per-row property**, not a global batch property.

Each row must satisfy:
- **All 10 queries succeed** → row succeeds with all 10 outputs
- **Any query fails** → row fails with `_error` marker, zero outputs

**This guarantee can be maintained even when processing multiple rows in parallel**, as long as:

1. **Results are grouped by row** after all queries complete
2. **Per-row failure checking** happens BEFORE emitting results
3. **Partial results are discarded** if any query in that row failed

### Implementation Pattern

```python
# 1. Execute all queries for all rows in parallel
all_results = executor.execute_batch(all_contexts, process_query)

# 2. Group results by row
rows_grouped = [all_results[i*10:(i+1)*10] for i in range(num_rows)]

# 3. Check atomicity PER ROW
for row_idx, row_results in enumerate(rows_grouped):
    failed = [r for r in row_results if r.status != "success"]

    if failed:
        # ANY query failed → entire row fails
        output_rows.append({**original_row, "_error": ...})
    else:
        # ALL queries succeeded → merge all outputs
        output = dict(original_row)
        for result in row_results:
            output.update(result.row)
        output_rows.append(output)
```

**Atomicity is preserved** because failure checking happens AFTER all queries complete but BEFORE emitting results.

### Stress Test Needed

We need to verify atomicity holds with:
- 100 rows × 10 queries = 1000 total queries
- pool_size=100 (all 1000 queries in flight simultaneously)
- 30% capacity error rate (300 queries fail and retry)
- Verify: No partial rows emitted (0 or 10 outputs, never 1-9)

---

## Root Cause Analysis

### Why Does This Code Exist?

The comment at line 402-408 suggests this was intentional:

> "This method uses ThreadPoolExecutor directly for per-row query parallelism rather than PooledExecutor.execute_batch(). The distinction:
>
> - PooledExecutor.execute_batch(): Designed for cross-row batching with AIMD throttling to adaptively manage rate limits across many rows.
> - ThreadPoolExecutor here: Simple parallel execution within a single row. All queries share the same underlying AzureOpenAI client which handles its own rate limiting, so AIMD overhead is unnecessary."

**This reasoning is flawed:**

1. **"AzureOpenAI client handles its own rate limiting"** - FALSE
   The Azure OpenAI SDK raises exceptions on rate limits, it does NOT retry.

2. **"AIMD overhead is unnecessary"** - FALSE
   AIMD is CRITICAL for production resilience. Without it, all queries fail on rate limit.

3. **"Per-row query parallelism"** vs **"cross-row batching"** - MISUNDERSTOOD
   PooledExecutor supports BOTH. You can parallelize across rows AND queries.

### Likely Origin

This code may have been written:
1. Before `PooledExecutor` existed (prototyping phase)
2. Based on a misunderstanding of how Azure OpenAI SDK handles rate limits
3. Optimized for simplicity over production resilience

---

## Impact Assessment

### On Previous Profiling Results

The profiling report I generated is **partially invalid**:

✅ **Still valid:**
- Atomicity guarantee verification (rows are atomic, just with wrong failure semantics)
- Client caching efficiency
- Batch processing overhead measurement
- Query timing distribution

❌ **Invalid:**
- "Capacity error handling" - Tested immediate failure, not retry behavior
- "Rate limit error handling" - Plugin doesn't handle it correctly
- Performance characteristics - Missing retry delays in timing

### On Production Deployment

**If deployed as-is:**
- ❌ Any rate limit → 100% of affected rows fail permanently
- ❌ Batch processing is 10x slower than it should be (for large pool sizes)
- ❌ Poor resilience to transient API issues

---

## Recommended Actions

### Immediate (CRITICAL)

1. **FIX ISSUE 1:** Refactor to use `PooledExecutor.execute_batch()` with retries
   - Remove ThreadPoolExecutor usage in `_execute_queries_parallel`
   - Flatten queries across all rows
   - Use PooledExecutor for retry logic

2. **FIX ISSUE 2:** Enable cross-row parallelism
   - Process multiple rows simultaneously (up to pool_size / queries_per_row)
   - Maintain atomicity by grouping results back by row

3. **UPDATE TESTS:** Rewrite profiling tests to verify:
   - Capacity errors trigger retries (not immediate failure)
   - AIMD backoff increases delay on repeated failures
   - Rows eventually succeed after transient rate limits pass
   - Full pool utilization with large pool sizes

### Short-term

1. Add integration test with real Azure OpenAI endpoint to verify retry behavior
2. Load test with concurrent row processing + capacity errors
3. Update documentation to reflect correct behavior

---

## Example: Correct Implementation Sketch

```python
def _process_batch(self, rows, ctx):
    """Process batch with full parallelism and retry support."""
    if not rows or self._executor is None:
        return self._process_batch_sequential(rows, ctx)  # Fallback

    # Flatten: (N rows × M queries) → single list of work items
    contexts = []
    for row_idx, row in enumerate(rows):
        for query_idx, spec in enumerate(self._query_specs):
            work_idx = row_idx * len(self._query_specs) + query_idx
            contexts.append(RowContext(
                row={"original": row, "spec": spec},
                state_id=f"{ctx.state_id}_r{row_idx}_q{query_idx}",
                row_index=work_idx,
            ))

    # Execute all queries with retry support
    all_results = self._executor.execute_batch(
        contexts=contexts,
        process_fn=lambda work, state_id: self._process_single_query(
            work["original"], work["spec"], state_id
        ),
    )

    # Group results back by row (M queries per row)
    M = len(self._query_specs)
    output_rows = []

    for row_idx, original_row in enumerate(rows):
        start = row_idx * M
        end = start + M
        row_results = all_results[start:end]

        # Check atomicity: all-or-nothing per row
        failed = [r for r in row_results if r.status != "success"]

        if failed:
            # ANY query failed → row fails
            error_row = dict(original_row)
            error_row["_error"] = {
                "reason": "query_failed",
                "failed_queries": [...],
            }
            output_rows.append(error_row)
        else:
            # ALL queries succeeded → merge outputs
            output = dict(original_row)
            for result in row_results:
                output.update(result.row)
            output_rows.append(output)

    return TransformResult.success_multi(output_rows)
```

---

## Conclusion

The Azure Multi-Query plugin has **two critical implementation gaps**:

1. **No retry logic** - Fails immediately on capacity errors (should retry with AIMD backoff)
2. **Sequential row processing** - Wastes pool capacity (should process multiple rows in parallel)

**Both issues can be fixed** by using `PooledExecutor.execute_batch()` correctly.

**Atomicity can still be maintained** even with concurrent row processing, as long as results are grouped by row and failure checking happens before emission.

**Previous profiling report is partially invalid** - needs to be re-run after fixes are implemented.

---

**Next Steps:**

1. Discuss fix approach with team
2. Implement PooledExecutor-based solution
3. Write new tests for retry behavior + concurrent row processing
4. Re-profile to verify performance improvements
5. Stress test atomicity under high concurrency + capacity errors

---

**Documents to Update:**
- `AZURE_MULTI_QUERY_PROFILING_REPORT.md` - Mark as "testing wrong implementation"
- `AZURE_MULTI_QUERY_ATOMICITY_VERIFICATION.md` - Atomicity logic is correct, but failure semantics are wrong
- `AZURE_MULTI_QUERY_OPTIMIZATIONS.md` - Retry logic is #1 priority, not pool_size tuning

**Generated by:** Claude Code (Sonnet 4.5)
**Date:** 2026-01-26
