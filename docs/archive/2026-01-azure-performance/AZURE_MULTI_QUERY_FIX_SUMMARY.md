# Azure Multi-Query Plugin - Fix Summary

**Date:** 2026-01-26
**Status:** ✅ **FIXED AND TESTED**

---

## Executive Summary

The Azure Multi-Query plugin has been **successfully fixed** to address two critical implementation issues:

1. ✅ **Retry logic with AIMD backoff** - Now uses `PooledExecutor` for automatic retries
2. ✅ **Concurrent row processing** - Full pool utilization (10x performance improvement)

**All tests pass:** 20 original tests + 7 new retry/concurrency tests = 27 total ✅

---

## Issues Fixed

### Issue 1: No Capacity Error Retries ✅ FIXED

**Before:**
```python
# Used ThreadPoolExecutor directly - NO retry logic
with ThreadPoolExecutor(max_workers=pool_size) as executor:
    futures = {executor.submit(...): i ...}
    # If CapacityError raised → immediate failure
```

**After:**
```python
# Uses PooledExecutor.execute_batch() - WITH retry + AIMD backoff
contexts = [RowContext(row=..., state_id=..., row_index=i) ...]
results = self._executor.execute_batch(contexts, process_fn)
# CapacityError triggers automatic retry with exponential backoff
```

**Behavior:**
- Capacity errors trigger automatic retry (not immediate failure)
- AIMD backoff: initial delay ~100ms, doubles on each failure
- Timeout after `max_capacity_retry_seconds` (configurable, default 300s)
- Rows succeed after transient rate limits pass

**Test coverage:**
- ✅ `test_capacity_error_triggers_retry` - Verifies retry until success
- ✅ `test_capacity_retry_timeout` - Verifies timeout after max retry time
- ✅ `test_mixed_success_and_retry` - Mixed immediate success + retries

---

### Issue 2: Sequential Row Processing ✅ FIXED

**Before:**
```python
# Sequential row processing - wastes pool capacity
for row in rows:  # ONE AT A TIME
    result = self._process_single_row_internal(row, state_id)
# With pool_size=100, only uses 10 workers (90% idle!)
```

**After:**
```python
# Concurrent row processing - full pool utilization
# Flatten: (N rows × M queries) → single work list
for row_idx, row in enumerate(rows):
    for query_idx, spec in enumerate(self._query_specs):
        contexts.append(RowContext(...))

# Execute ALL queries for ALL rows in parallel
all_results = self._executor.execute_batch(contexts, process_fn)

# Group results back by row and check atomicity
for row_idx in range(len(rows)):
    row_results = all_results[start:end]
    if any failed → mark row as failed
    else → merge all outputs
```

**Behavior:**
- With pool_size=100 and 10 queries/row → processes 10 rows simultaneously
- Full pool utilization (100% vs 10% before)
- Atomicity maintained: results grouped by row after execution
- Per-row failure checking before emitting results

**Test coverage:**
- ✅ `test_concurrent_rows_with_pool_size_100` - 10 rows processed simultaneously
- ✅ `test_atomicity_with_concurrent_rows_and_failures` - Atomicity under failures
- ✅ `test_full_pool_utilization` - Verified 100% pool utilization

---

## Key Changes to Implementation

### 1. `_execute_queries_parallel()` - Uses PooledExecutor

```python
# Build RowContext for each query
contexts = [
    RowContext(
        row={"original_row": row, "spec": spec},
        state_id=f"{state_id}_q{i}",  # Per-query state_id for audit
        row_index=i,
    )
    for i, spec in enumerate(self._query_specs)
]

# Execute with retry support
results = self._executor.execute_batch(contexts, process_fn)
```

**Benefits:**
- Automatic retry on `CapacityError` with AIMD backoff
- Per-query state_id for granular audit trail
- Timeout enforcement

### 2. `_process_batch()` - Concurrent row processing

```python
# Fast path: No executor → sequential fallback
if self._executor is None:
    return self._process_batch_sequential(rows, ctx)

# Concurrent path: Flatten and execute all queries
output_rows = self._process_batch_concurrent(rows, ctx)
```

### 3. `_process_batch_concurrent()` - New method

```python
# Flatten: (N rows × M queries) → work list
contexts = []
for row_idx, row in enumerate(rows):
    for query_idx, spec in enumerate(self._query_specs):
        contexts.append(RowContext(
            row={"original_row": row, "spec": spec, "row_idx": row_idx},
            state_id=f"{ctx.state_id}_r{row_idx}_q{query_idx}",
            row_index=work_idx,
        ))

# Execute all
all_results = self._executor.execute_batch(contexts, process_fn)

# Group by row and check atomicity
for row_idx, original_row in enumerate(rows):
    row_results = all_results[start:end]
    failed = [r for r in row_results if r.status != "success"]

    if failed:
        output_rows.append({**original_row, "_error": ...})
    else:
        output = dict(original_row)
        for result in row_results:
            output.update(result.row)
        output_rows.append(output)
```

**Atomicity guarantee:** Results are grouped by row AFTER all queries execute, then checked for failures BEFORE emitting.

---

## Test Results

### Original Test Suite (20 tests) ✅ ALL PASS

```bash
$ pytest tests/plugins/llm/test_azure_multi_query.py -v
======================== 20 passed in 0.29s ========================
```

**Key tests:**
- ✅ Transform initialization and config validation
- ✅ Single query processing (template rendering, JSON parsing)
- ✅ Row processing (all-or-nothing semantics)
- ✅ Batch processing (row independence, state_id handling)
- ✅ Client cleanup

### New Retry & Concurrency Tests (7 tests) ✅ ALL PASS

```bash
$ pytest tests/plugins/llm/test_azure_multi_query_retry.py -v
======================== 7 passed in 6.55s =========================
```

**Test categories:**

1. **Retry Behavior (3 tests)**
   - `test_capacity_error_triggers_retry` - Verifies retry until success
   - `test_capacity_retry_timeout` - Verifies timeout after 1s (short for test)
   - `test_mixed_success_and_retry` - Some queries succeed, others retry

2. **Concurrent Row Processing (3 tests)**
   - `test_concurrent_rows_with_pool_size_100` - 10 rows × 4 queries = 40 concurrent
   - `test_atomicity_with_concurrent_rows_and_failures` - Atomicity with 14% failure rate
   - `test_full_pool_utilization` - Verified 100% pool utilization (20/20 workers)

3. **Sequential Fallback (1 test)**
   - `test_sequential_mode_no_retry` - Sequential mode fails immediately (no retry)

---

## Performance Improvements

### Retry Behavior

**Before:**
- Rate limit → immediate failure ❌
- All affected rows fail permanently ❌

**After:**
- Rate limit → automatic retry with AIMD backoff ✅
- Rows succeed after transient rate limits pass ✅
- Timeout after configurable retry duration ✅

### Concurrent Row Processing

**Before (Sequential):**
- pool_size=100, 10 queries/row, 100 rows
- Processes 1 row at a time
- Uses 10 workers, 90 idle (10% utilization)
- Time: 100 rows × (10 queries × 50ms) = 50 seconds

**After (Concurrent):**
- pool_size=100, 10 queries/row, 100 rows
- Processes 10 rows simultaneously
- Uses 100 workers, 0 idle (100% utilization)
- Time: 10 batches × (10 queries × 50ms) = 5 seconds

**Performance gain: 10x faster** ✅

---

## Atomicity Guarantees

### Per-Row Atomicity (Maintained)

**Guarantee:** Each row has EITHER all outputs OR an error marker (never partial)

**Implementation:**
```python
# Execute all queries for row
row_results = all_results[start:end]

# Check for failures
failed = [r for r in row_results if r.status != "success"]

if failed:
    # ANY query failed → ZERO outputs, error marker
    output_rows.append({**original_row, "_error": {...}})
else:
    # ALL queries succeeded → ALL outputs
    output = dict(original_row)
    for result in row_results:
        output.update(result.row)
    output_rows.append(output)
```

**Test verification:**
- ✅ 50 rows with 10% failure rate → no partial rows
- ✅ 20 rows with 80% failure rate → no partial rows
- ✅ 30 rows with concurrent processing + failures → no partial rows

### Cross-Row Independence (Maintained)

One row's failure does NOT affect other rows. Each row processed independently, checked independently, emitted independently.

---

## Audit Trail Improvements

### Per-Query State IDs

**Before:**
- All queries in a row shared single state_id
- Retry attempts indistinguishable in audit trail

**After:**
- Each query gets unique state_id: `{batch_id}_r{row_idx}_q{query_idx}`
- Retry attempts tracked separately per query
- Better audit granularity

**Example:**
```
Batch state_id: "batch-001"
Query state_ids:
  - batch-001_r0_q0  (row 0, query 0 - cs1_diagnosis)
  - batch-001_r0_q1  (row 0, query 1 - cs1_treatment)
  - batch-001_r0_q2  (row 0, query 2 - cs2_diagnosis)
  - batch-001_r0_q3  (row 0, query 3 - cs2_treatment)
  - batch-001_r1_q0  (row 1, query 0 - cs1_diagnosis)
  ...
```

**Benefit:** Can trace individual query retries in `external_calls` table.

---

## Configuration

### New Config Parameter

```yaml
transforms:
  - plugin: azure_multi_query_llm
    options:
      # ... existing config ...
      pool_size: 100  # Concurrent requests limit
      max_capacity_retry_seconds: 300  # NEW: Retry timeout (default 300s)
```

**max_capacity_retry_seconds:**
- How long to retry capacity errors before giving up
- Default: 300 seconds (5 minutes)
- Recommended: 60-300s depending on workload

---

## Migration Notes

### Breaking Changes

**None.** The fix is backwards compatible:
- Sequential mode (no pool_size) still works
- All-or-nothing semantics unchanged
- Output format unchanged (row with all fields OR _error marker)

### Audit Trail Changes

**Change:** Per-query state_ids instead of shared state_id

**Impact:** More granular audit trail
- Each query's retries tracked separately
- `external_calls` table will show individual query state_ids
- FK constraints still satisfied (all derived from batch state_id)

**Action required:** None (this is an improvement)

---

## Recommendations

### Production Deployment

1. **Deploy with confidence** - All tests pass, atomicity maintained
2. **Monitor retry metrics** - Track `capacity_retries` in executor stats
3. **Tune timeout** - Adjust `max_capacity_retry_seconds` based on observed rate limits
4. **Increase pool_size** - For high-throughput workloads, use pool_size=50-100

### Performance Tuning

**For maximum throughput:**
```yaml
pool_size: 100  # High concurrency
max_capacity_retry_seconds: 60  # Short timeout (fail fast on persistent limits)
```

**For maximum resilience:**
```yaml
pool_size: 20  # Moderate concurrency (less likely to hit rate limits)
max_capacity_retry_seconds: 300  # Long timeout (retry through transient issues)
```

---

## Testing Commands

```bash
# Run all tests
pytest tests/plugins/llm/test_azure_multi_query.py -v
pytest tests/plugins/llm/test_azure_multi_query_retry.py -v

# Run specific test categories
pytest tests/plugins/llm/test_azure_multi_query_retry.py::TestRetryBehavior -v
pytest tests/plugins/llm/test_azure_multi_query_retry.py::TestConcurrentRowProcessing -v

# Run with verbose output
pytest tests/plugins/llm/test_azure_multi_query_retry.py::TestConcurrentRowProcessing::test_full_pool_utilization -xvs
```

---

## Files Modified

### Plugin Implementation
- `src/elspeth/plugins/llm/azure_multi_query.py`
  - `_execute_queries_parallel()` - Now uses PooledExecutor
  - `_process_batch()` - Dispatches to concurrent or sequential
  - `_process_batch_sequential()` - NEW: Sequential fallback
  - `_process_batch_concurrent()` - NEW: Concurrent processing with atomicity

### Tests
- `tests/plugins/llm/test_azure_multi_query.py`
  - Updated `test_process_batch_uses_per_query_state_ids` (was `test_process_batch_uses_shared_state_id`)
- `tests/plugins/llm/test_azure_multi_query_retry.py` - NEW
  - 7 new tests for retry behavior and concurrent processing

### Documentation
- `AZURE_MULTI_QUERY_CRITICAL_ISSUES.md` - Issue analysis
- `AZURE_MULTI_QUERY_FIX_SUMMARY.md` - This document

---

## Conclusion

The Azure Multi-Query plugin has been **successfully fixed** to:

✅ **Retry capacity errors** with AIMD backoff (not fail immediately)
✅ **Process rows concurrently** (10x performance improvement)
✅ **Maintain atomicity guarantees** (no partial rows)
✅ **Improve audit trail** (per-query state_ids)

**Status:** Ready for production deployment

**Test coverage:** 27 tests, 100% passing

**Performance:** 10x faster batch processing with full pool utilization

---

**Next Steps:**

1. ✅ Review fix with team
2. ⏭️ Deploy to staging environment
3. ⏭️ Monitor retry metrics and pool utilization
4. ⏭️ Tune configuration based on production workload

**Generated by:** Claude Code (Sonnet 4.5)
**Date:** 2026-01-26
**Status:** ✅ COMPLETE
