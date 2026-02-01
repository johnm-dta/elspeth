# Test Quality Review: test_retry_integration.py

## Summary

Integration test suite verifies retry audit trail completeness but has critical gaps around backoff metadata capture (per CLAUDE.md requirement), missing edge cases for audit integrity, and infrastructure deficiencies. Tests correctly verify attempt recording but don't validate the complete audit contract.

## Poorly Constructed Tests

### Test: test_each_retry_attempt_recorded_as_separate_node_state (line 161)
**Issue**: Missing verification of backoff metadata capture
**Evidence**: CLAUDE.md states "Backoff metadata captured" as non-negotiable retry semantic. Test verifies attempts, statuses, token_ids but never checks:
- `duration_ms` between attempts (should show exponential backoff)
- `started_at` timestamps (should verify delay occurred)
- `completed_at` timestamps (should show execution time)
- No assertion that backoff delay actually happened

```python
# Current: Only checks attempt numbers and status
attempts = [row.attempt for row in rows]
assert attempts == [0, 1, 2]
statuses = [row.status for row in rows]

# Missing: Backoff timing verification
# Should verify started_at[1] - completed_at[0] >= base_delay
# Should verify started_at[2] - completed_at[1] >= base_delay * exponential_base
```

**Fix**: Add timing assertions to verify backoff metadata:
```python
# Verify backoff delays were applied
for i in range(1, len(rows)):
    prev_completed = rows[i-1].completed_at
    curr_started = rows[i].started_at
    delay = (curr_started - prev_completed).total_seconds()
    # Should be >= base_delay (accounting for jitter)
    assert delay >= 0.001, f"Backoff delay missing between attempt {i-1} and {i}"
```
**Priority**: P0 - Core audit requirement violation

### Test: test_each_retry_attempt_recorded_as_separate_node_state (line 161)
**Issue**: No verification of input_hash stability across retries
**Evidence**: For audit integrity, the same row being retried must have identical input_hash across all attempts. Test doesn't verify this critical invariant:
```python
# Missing verification
input_hashes = [row.input_hash for row in rows]
assert len(set(input_hashes)) == 1, "Input hash must be identical across retries"
```
**Fix**: Add input hash stability check
**Priority**: P0 - Audit integrity requirement

### Test: test_max_retries_exceeded_all_attempts_recorded (line 261)
**Issue**: Doesn't verify error_json contains retryable classification
**Evidence**: Test checks `error_json is not None` but doesn't validate:
- Error type is preserved (ConnectionError)
- Error message is captured
- Retryable flag or classification is recorded

```python
# Current: Minimal check
assert all(e is not None for e in errors)

# Missing: Error content validation
import json
for error in errors:
    error_data = json.loads(error)
    assert "ConnectionError" in error_data.get("type", "")
    assert "attempt" in error_data  # Which attempt failed?
```
**Fix**: Parse and validate error_json structure
**Priority**: P1 - Audit completeness gap

### Test: test_single_attempt_no_retry_records_single_node_state (line 367)
**Issue**: Incomplete test - name promises verification but doesn't check retry manager wasn't involved
**Evidence**: Test bypasses retry manager entirely (`execute_transform` called directly), so it's not actually testing "single attempt no retry" behavior - it's testing "no retry manager at all". Should use RetryManager with max_attempts=1 to verify the single-attempt path through retry infrastructure.
**Fix**: Execute through RetryManager with max_attempts=1 to test the actual single-attempt code path
**Priority**: P2 - Test doesn't match its claimed scope

### Test: test_max_retries_exceeded_all_attempts_recorded (line 261)
**Issue**: Doesn't verify that MaxRetriesExceeded preserves full attempt history
**Evidence**: Test catches `MaxRetriesExceeded` and checks `attempts == 2` but doesn't verify the exception allows reconstruction of what happened:
```python
# Missing: Can we reconstruct attempt history from exception?
assert hasattr(exc_info.value, 'last_error')
assert str(exc_info.value.last_error) == f"Permanent failure attempt 2"
# Does the audit trail have everything needed to explain this to an auditor?
```
**Fix**: Add assertions that exception + audit trail provide complete failure story
**Priority**: P1 - Auditability requirement

## Misclassified Tests

### Test: All tests in TestRetryAuditTrail
**Issue**: Tests are correctly classified as integration but have unit-test-level mocking
**Evidence**: Uses `Mock(spec=SpanFactory)` for span_factory instead of real SpanFactory. Integration tests should use real components or explain why mocking is necessary. The span recording is part of the audit trail - mocking it defeats the purpose of integration testing the full audit chain.
```python
span_factory = Mock(spec=SpanFactory)
span_factory.transform_span.return_value.__enter__ = Mock(return_value=None)
```
**Fix**: Either:
1. Use real SpanFactory with in-memory tracer backend
2. Document why SpanFactory must be mocked (if it has external dependencies)
3. Consider extracting a "RetryAuditContract" unit test suite that can use mocks
**Priority**: P2 - Architectural boundary question

## Infrastructure Gaps

### Gap: Repeated manual attempt tracking across all tests
**Issue**: Every test reimplements attempt tracking with `attempt_tracker = {"current": 0}` and manual increment
**Evidence**: Lines 211-222, 309-321 - identical pattern repeated
```python
attempt_tracker = {"current": 0}

def execute_attempt() -> tuple[TransformResult, TokenInfo, str | None]:
    attempt = attempt_tracker["current"]
    attempt_tracker["current"] += 1
    return transform_executor.execute_transform(...)
```
**Fix**: Extract to helper method or fixture:
```python
def make_tracked_executor(self, transform_executor, transform, token, ctx, step):
    """Returns (execute_fn, get_attempt_count_fn)."""
    attempt = 0
    def execute():
        nonlocal attempt
        result = transform_executor.execute_transform(
            transform=transform, token=token, ctx=ctx,
            step_in_pipeline=step, attempt=attempt
        )
        attempt += 1
        return result
    return execute, lambda: attempt
```
**Priority**: P3 - Code duplication, maintenance burden

### Gap: No fixture for common test environment setup
**Issue**: Every test method calls `self._setup_run_and_node()` and `self._create_token()` manually
**Evidence**: Lines 178-195, 278-295, 381-398 - identical setup pattern
**Fix**: Create pytest fixture that returns pre-configured environment:
```python
@pytest.fixture
def retry_test_context(self, test_env):
    """Returns (run_id, node_id, source_node_id, make_token_fn)."""
    # One-time setup for retry testing
    recorder = test_env["recorder"]
    run_id, node_id = self._setup_run_and_node(recorder)
    # ... source node setup
    def make_token(row_data):
        return self._create_token(recorder, run_id, source_node_id, row_data)
    return run_id, node_id, make_token
```
**Priority**: P3 - Reduces boilerplate, improves test clarity

### Gap: No test for partial retry scenario (some succeed, some fail)
**Issue**: Missing test case for realistic scenario where a batch of rows has mixed retry behavior
**Evidence**: All tests process single row. Real pipelines process many rows where:
- Row A succeeds immediately
- Row B fails once, succeeds on retry
- Row C exhausts retries
- Audit trail must distinguish all three
**Fix**: Add test `test_mixed_retry_outcomes_in_batch` that verifies audit trail correctness for multiple rows with different retry paths
**Priority**: P2 - Missing realistic scenario

### Gap: No test for non-retryable error path
**Issue**: Tests only cover retryable (ConnectionError) and success paths
**Evidence**: No test verifies that non-retryable errors (e.g., ValueError) are recorded correctly:
- Should only have attempt=0 record
- Should have status="failed"
- Should NOT retry
**Fix**: Add test `test_non_retryable_error_single_attempt` that raises TypeError or ValueError and verifies single failed attempt recorded
**Priority**: P1 - Missing critical error path

### Gap: No test for uniqueness constraint validation
**Issue**: Schema has `UniqueConstraint("token_id", "node_id", "attempt")` but no test verifies this is enforced
**Evidence**: Lines 186 in schema.py - constraint exists but integration test doesn't verify the database enforces it
**Fix**: Add test that attempts to record duplicate (token_id, node_id, attempt) and expects IntegrityError
**Priority**: P2 - Database contract not verified

### Gap: No test for context_before_json and context_after_json across retries
**Issue**: Tests don't verify that context is captured for each attempt
**Evidence**: CLAUDE.md requires "Transform boundaries - Input AND output captured at every transform". For retries, this means context_before/after for EACH attempt, not just the final one.
**Fix**: Add assertions:
```python
for row in rows:
    assert row.context_before_json is not None
    # For failed attempts, context_after_json might be None
    if row.status == "completed":
        assert row.context_after_json is not None
```
**Priority**: P1 - Audit completeness gap

## Positive Observations

- Tests correctly verify attempt numbering (0, 1, 2...)
- Tests properly use RetryManager and validate MaxRetriesExceeded exception
- Tests verify all attempts are recorded (no silent drops)
- Helper methods `_setup_run_and_node` and `_create_token` provide clean abstraction
- Tests use in-memory database (fast, isolated)
- Test docstrings clearly explain what's being verified
- FlakyTransform and AlwaysFailTransform are well-designed test doubles
- Tests verify both success-after-retry and exhaustion paths

## Risk Assessment

**Current State**: Tests prove retry attempts are recorded but don't verify the full audit contract per CLAUDE.md.

**High-Stakes Auditability Impact**:
1. **Missing backoff metadata**: Auditor asks "why did the system wait 4 seconds between attempts?" - we can't answer because timing isn't verified
2. **Missing error detail validation**: Auditor asks "what exactly failed on attempt 1?" - error_json structure is untested
3. **Missing non-retryable path**: Auditor asks "how do we know this error wasn't retried when it shouldn't have been?" - no test coverage
4. **Missing context capture**: Auditor asks "what was the transform's input state on attempt 2?" - context_before_json not verified

**Recommendation**: Before RC-1 release, add P0/P1 tests. The current suite proves the basic mechanism works but doesn't verify the audit trail meets the "I don't know what happened is never acceptable" standard.

## Confidence Assessment

- ✅ **High confidence**: Attempt recording mechanism works (tests prove this)
- ⚠️ **Medium confidence**: Backoff metadata is captured (not tested but schema has duration_ms/timestamps)
- ❌ **Low confidence**: Error details are complete (error_json is tested for existence, not structure)
- ❌ **Low confidence**: Non-retryable errors are handled correctly (no test coverage)

## Information Gaps

To complete review, need:
1. RetryManager backoff calculation logic (should verify exponential with jitter is applied)
2. Error serialization format (to validate error_json structure)
3. RowProcessor integration (to understand how attempt tracking flows end-to-end)
4. Production retry policies (to ensure test coverage matches real-world configuration)

## Caveats

- Review based on CLAUDE.md requirements and SME Agent Protocol standards
- Did not execute tests (static analysis only)
- Did not review related unit tests in `test_retry.py` (out of scope)
- Assumed schema in `node_states_table` is final (migrations pending)
