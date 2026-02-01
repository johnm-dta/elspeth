# Test Quality Review: test_pooled_executor.py

## Summary
Test suite has strong concurrency/threading tests but suffers from pervasive sleepy assertions (20+ instances), inadequate audit trail verification, missing edge cases, and zero property-based testing. The suite tests infrastructure behavior (threading, semaphores) but never validates the core audit requirement: "Full request AND response recorded" from CLAUDE.md.

## Poorly Constructed Tests

### Test: test_execute_batch_returns_results_in_order (line 82)
**Issue**: Sleepy assertion - uses fixed 0.01s delays to force ordering
**Evidence**: `time.sleep(0.01 * (3 - idx))` to artificially create out-of-order completion
**Fix**: Use threading events and barriers instead of sleep. Test actual concurrent completion with unpredictable timing, verify reorder buffer handles it correctly.
**Priority**: P1

### Test: test_execute_batch_respects_pool_size (line 146)
**Issue**: Sleepy assertion - uses fixed 0.05s sleep to measure concurrency
**Evidence**: `time.sleep(0.05)` inside mock_process. If system is slow, test could false-pass with >pool_size concurrency.
**Fix**: Use atomic operations and barriers. Capture exact moment of semaphore acquisition/release. Verify semaphore._value never goes negative or exceeds pool_size.
**Priority**: P1

### Test: test_capacity_retry_releases_semaphore_during_backoff (line 310)
**Issue**: Complex sleepy assertion with threading events, but still relies on `time.sleep(0.05)` for timing
**Evidence**: Line 350: `time.sleep(0.05)  # Give row 0 time to release semaphore`
**Fix**: Replace sleep with explicit event signaling from semaphore release point. Test deadlock scenario directly: submit pool_size+1 rows where first pool_size all hit capacity errors - if semaphore isn't released, row N+1 never starts.
**Priority**: P0 (Critical correctness property - deadlock prevention)

### Test: test_no_deadlock_when_batch_exceeds_pool_with_capacity_errors (line 378)
**Issue**: Uses SIGALRM timeout as test mechanism - fragile on different systems/CI
**Evidence**: Lines 420-431 set up signal handlers for timeout detection
**Fix**: Use threading.Event with timeout or concurrent.futures.wait(timeout=...). Signals are process-global and can interfere with other tests. Also, 10 second timeout means this test takes 10s if it fails.
**Priority**: P2

### Test: test_capacity_error_triggers_throttle_and_retries (line 223)
**Issue**: Only tests single capacity error then success - doesn't test multiple retries with escalating delays
**Evidence**: `current_count == 1` condition means only one retry occurs
**Fix**: Test sequence: fail, retry (verify delay applied), fail again (verify delay increased via AIMD), succeed. Verify throttle stats show correct backoff progression.
**Priority**: P1

### Test: test_shutdown_completes_pending (line 46)
**Issue**: Doesn't actually test pending completion - submits zero work, calls shutdown
**Evidence**: No `execute_batch` call before shutdown. Test verifies shutdown doesn't crash, not that it waits for work.
**Fix**: Submit slow batch, call shutdown(wait=True) from another thread, verify batch completes before shutdown returns. Test shutdown(wait=False) interrupts work.
**Priority**: P2

### Test: test_row_context_immutable_reference (line 69)
**Issue**: Tests mutation vulnerability but doesn't flag it as a problem - just documents current behavior
**Evidence**: Comment says "shared reference" as if this is acceptable. Per CLAUDE.md prohibition on defensive patterns, if RowContext can be mutated after creation, that's a bug.
**Fix**: Either make RowContext freeze the row dict (deep copy) or document that caller MUST NOT mutate. Add test that verifies executor doesn't mutate row data. This is critical for audit integrity - if row is mutated mid-execution, audit trail is corrupted.
**Priority**: P0 (Audit integrity)

## Missing Critical Tests

### No Audit Trail Verification
**Issue**: Zero tests verify that process_fn calls are recorded with full request/response
**Evidence**: All tests use mock_process functions that return results, but none verify the results are suitable for audit trail recording. No tests check that state_id is preserved correctly through retries.
**Fix**: Add tests that verify:
1. Every process_fn call gets a unique audit record (via state_id + call_index)
2. Capacity error retries create SEPARATE audit records for each attempt
3. Results contain enough context to explain "why did this row get this result"
4. Stats returned by get_stats() match reality (e.g., capacity_retries count is accurate)
**Priority**: P0 (Core auditability requirement from CLAUDE.md)

### No Exception Safety Tests
**Issue**: What happens if process_fn raises unexpected exception (not CapacityError)?
**Evidence**: No tests for ValueError, RuntimeError, network errors, or other unexpected failures
**Fix**: Test that non-CapacityError exceptions are:
1. Caught and converted to TransformResult.error (not propagated)
2. Return with retryable=False
3. Don't leak semaphore (verify pending_count returns to 0)
4. Don't corrupt reorder buffer (subsequent batches work correctly)
**Priority**: P1

### No Empty/Degenerate Batch Tests
**Issue**: Only one test (implicit) checks empty batch - no tests for single-row batch, all-failures, all-capacity-errors-timeout
**Evidence**: test_execute_batch with empty contexts returns early (line 163-164), but no explicit test
**Fix**: Add tests for:
1. Empty batch → returns []
2. Single row → no concurrency, works correctly
3. All rows fail (non-capacity errors) → all results are error status
4. All rows timeout on capacity retries → all results are capacity_retry_timeout errors
**Priority**: P2

### No Capacity Retry Timeout Edge Cases
**Issue**: test_capacity_retry_respects_max_timeout only tests "always fail" - doesn't test timeout mid-retry
**Evidence**: mock_process always raises CapacityError. Doesn't test case where timeout expires DURING a retry attempt (not between attempts).
**Fix**: Test that if timeout expires while process_fn is executing, the result is still returned (timeout only applies between retries, not during execution). Verify elapsed_seconds in error result is accurate.
**Priority**: P2

### No Semaphore Leak Tests
**Issue**: No direct verification that semaphore is never leaked in error paths
**Evidence**: Tests assume semaphore is correct, don't verify _semaphore._value
**Fix**: After every test that involves errors (capacity, timeout, exception), verify:
1. `executor.pending_count == 0`
2. `executor._semaphore._value == pool_size` (all permits released)
3. Submit another batch - should work correctly (no starvation from leaked permits)
**Priority**: P1

### No Concurrent Batch Tests
**Issue**: All tests submit one batch at a time - no tests for concurrent execute_batch calls from multiple threads
**Evidence**: Single-threaded test pattern throughout
**Fix**: Test that multiple threads can call execute_batch concurrently and:
1. Results are correctly isolated (batch A results don't leak into batch B)
2. Semaphore is shared correctly (total concurrency ≤ pool_size)
3. Reorder buffer is thread-safe (no data races)
**Priority**: P1 (Thread safety is critical for pooled execution)

### No Stats Accuracy Tests
**Issue**: Only two tests check get_stats() - don't verify stats match actual execution
**Evidence**: test_get_stats_returns_pool_config and test_get_stats_includes_throttle_stats just check keys exist
**Fix**: Test that stats are accurate:
1. Submit 10 rows with 3 capacity errors → stats show exactly 3 capacity_retries
2. Track total_throttle_time_ms - verify it matches sum of all sleep times
3. Verify peak_delay_ms is updated correctly during AIMD backoff
**Priority**: P2

### No ReorderBuffer Edge Case Tests
**Issue**: ReorderBuffer is tested indirectly but never directly for edge cases
**Evidence**: No standalone tests for buffer behavior when results arrive in reverse order, with gaps, etc.
**Fix**: This should be in a separate test_reorder_buffer.py file, but flag that current tests don't verify:
1. Results arriving in reverse order (N, N-1, N-2, ..., 0)
2. Large gap (complete 0, 2, 4, 6, then 1, 3, 5)
3. Buffer correctness when used concurrently (thread safety)
**Priority**: P3 (Infrastructure gap - not critical if ReorderBuffer has its own tests)

## Misclassified Tests

### Test: test_row_context_creation (line 60)
**Issue**: This is a dataclass smoke test, not a unit test
**Evidence**: Just verifies RowContext.__init__ works - no behavior being tested
**Fix**: Delete this test. RowContext is a dataclass with no logic. If the dataclass definition is wrong, type checker will catch it. This is noise.
**Priority**: P3

### Test: test_creates_with_config (line 16)
**Issue**: Borderline - tests initialization but immediately shuts down without using the executor
**Evidence**: Creates executor, asserts two properties, shuts down
**Fix**: Merge with test_shutdown_completes_pending - initialize, submit work, verify shutdown. Don't test init in isolation.
**Priority**: P3

### Test: test_creates_throttle_from_config (line 27)
**Issue**: White-box test - reaches into _throttle internals
**Evidence**: Asserts on `executor._throttle.config.backoff_multiplier`
**Fix**: Test behavior, not internals. Submit batch with capacity errors, verify backoff delay increases by expected factor (via timing or stats). If backoff_multiplier is wrong, the behavior test will fail.
**Priority**: P2

## Infrastructure Gaps

### Missing Fixtures
**Issue**: Every test creates its own executor with shutdown in try/finally or bare shutdown() call
**Evidence**: Lines 25, 40, 54, 114, 144, 181, 200, 217, 255, 289, 308, 376, 445 all have `executor.shutdown()`
**Fix**: Create pytest fixture:
```python
@pytest.fixture
def executor(request):
    config = getattr(request, 'param', PoolConfig(pool_size=2))
    ex = PooledExecutor(config)
    yield ex
    ex.shutdown(wait=True)
```
Use `@pytest.mark.parametrize('executor', [PoolConfig(pool_size=X)], indirect=True)` for custom config.
**Priority**: P2

### Missing Mock Helpers
**Issue**: Every test defines its own mock_process with lock and call tracking - lots of duplication
**Evidence**: Lines 96-102, 128-131, 160-172, 231-240, 269-274, 296-298, 331-354, 404-413 all define similar mock functions
**Fix**: Create reusable mock builders:
```python
def make_tracked_process(behavior: Callable[[int, int], TransformResult]) -> tuple[Callable, list]:
    """Returns (mock_process, call_log) where call_log tracks (idx, attempt) tuples."""

def make_delayed_process(delays_by_idx: dict[int, float]) -> Callable:
    """Returns mock_process with specified delays per row index."""

def make_failing_process(fail_counts: dict[int, int]) -> Callable:
    """Returns mock_process that fails first N attempts for each row."""
```
**Priority**: P2

### No Property-Based Testing
**Issue**: All tests use small fixed batches (3-6 rows) - no generative testing of batch sizes, pool sizes, error patterns
**Evidence**: Hypothesis is in tech stack (CLAUDE.md line 310) but never used
**Fix**: Add property tests:
1. `@given(batch_size=st.integers(0, 100), pool_size=st.integers(1, 10))` - verify results.length == batch_size, order preserved
2. `@given(failure_pattern=st.lists(st.booleans()))` - verify retry behavior correct for arbitrary error patterns
3. `@given(capacity_errors=st.lists(st.integers(0, 5)))` - verify AIMD throttle stats match actual retry counts
**Priority**: P1 (Hypothesis is already in stack, should be used)

### No Concurrency Stress Tests
**Issue**: All concurrency tests use tiny batches (3-6 rows) and small pool sizes (2-4) - doesn't stress semaphore or reorder buffer
**Evidence**: Largest test is 6 rows with pool_size=2 (line 416)
**Fix**: Add stress tests (maybe marked `@pytest.mark.slow`):
1. 1000 rows, pool_size=10, random capacity errors - verify all complete correctly
2. 100 rows, pool_size=50 (over-provisioned) - verify no deadlock
3. 500 rows with 50% capacity errors - verify throttle prevents runaway retries
**Priority**: P2

### No Performance Regression Tests
**Issue**: No tests verify that parallelism actually speeds things up
**Evidence**: Tests verify correctness, not performance
**Fix**: Add benchmark test:
1. Submit 20 rows with 0.1s delay each, pool_size=1 → should take ~2s
2. Submit 20 rows with 0.1s delay each, pool_size=10 → should take ~0.2s
3. If serial time ≈ parallel time, parallelism is broken
Mark as `@pytest.mark.benchmark` and skip in CI if too slow.
**Priority**: P3 (Nice to have - functional correctness is more critical)

## Positive Observations

1. **Strong threading test coverage** - test_capacity_retry_releases_semaphore_during_backoff and test_no_deadlock_when_batch_exceeds_pool_with_capacity_errors are excellent regression tests for subtle concurrency bugs
2. **Good use of atomic tracking** - Most tests use Lock correctly to track call counts and execution order without data races
3. **Tests verify the critical invariant** - Results are in submission order (test_execute_batch_returns_results_in_order)
4. **Decent error path coverage** - Tests cover capacity errors, timeouts, and normal errors
5. **Tests are independent** - Each test creates its own executor and shuts down cleanly

## Recommended Fixes (Priority Order)

### P0 (Fix Before RC-1 Release)
1. Add audit trail verification tests - verify state_id, call_index, retry attempts are correctly tracked
2. Fix test_row_context_immutable_reference - either document that row mutation is forbidden or deep-copy in RowContext
3. Fix test_capacity_retry_releases_semaphore_during_backoff - replace sleep with events

### P1 (Fix Soon)
1. Replace all sleepy assertions with event-based synchronization
2. Add exception safety tests (non-CapacityError exceptions)
3. Add semaphore leak verification to error path tests
4. Add concurrent batch execution tests (thread safety)
5. Add property-based tests using Hypothesis

### P2 (Technical Debt)
1. Create pytest fixtures to eliminate shutdown() duplication
2. Create mock helpers to eliminate mock_process duplication
3. Test behavior instead of internals (white-box tests)
4. Add empty/degenerate batch tests
5. Add stats accuracy verification tests

### P3 (Nice to Have)
1. Delete trivial dataclass tests (test_row_context_creation)
2. Add concurrency stress tests
3. Add ReorderBuffer edge case tests (or verify they exist elsewhere)
4. Add performance regression tests
