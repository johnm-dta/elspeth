# Test Quality Review: test_replayer.py

## Summary

The test suite for `CallReplayer` is well-structured with good isolation and coverage of core functionality. However, it has critical gaps in testing Tier 1 trust violations (audit trail corruption), missing integration tests that use real database fixtures, and lacks property-based testing for replay determinism. The tests properly verify basic replay mechanics but don't stress-test the "audit trail must be pristine" principle.

## Poorly Constructed Tests

### Test: test_replay_with_none_response_data (line 255)
**Issue**: Test verifies defensive behavior but doesn't verify the error happens during construction vs. retrieval
**Evidence**: Test mocks `get_call_response_data.return_value = None` but doesn't verify the error message indicates which phase failed (lookup vs. payload retrieval)
**Fix**: Assert on specific error message content to ensure it mentions "payload purged" and includes the call_id for debugging
**Priority**: P3

### Test: test_replay_caches_results (line 155)
**Issue**: Cache test doesn't verify cache invalidation mutation vulnerabilities
**Evidence**: Test populates cache, but doesn't verify that mutating returned `response_data` doesn't pollute the cache for subsequent calls
**Fix**: Add assertion that modifying `result1.response_data` doesn't affect `result2.response_data` (cache should return independent copies, not references)
**Priority**: P2

### Test: test_different_call_types_same_hash_are_cached_separately (line 310)
**Issue**: Test uses side_effect functions but doesn't verify the call happened with exact parameters
**Evidence**: Uses `side_effect` with conditional logic, making it hard to verify the mock was called with correct args. The side_effect could silently pass even if call_type parameter is wrong.
**Fix**: Use `assert_called_with` for both database calls to verify exact parameters, not just call_count
**Priority**: P2

## Missing Critical Tests

### Test: Audit trail corruption scenarios (MISSING)
**Issue**: No tests verify crash-on-corrupt-data behavior for Tier 1 trust violations
**Evidence**: CLAUDE.md states "Bad data in the audit trail = crash immediately" - no tests verify this for:
- NULL `call_id` from database
- Invalid `CallStatus` enum value from database
- NULL `request_hash` from database
- Wrong type for `latency_ms` (string instead of float)
- Malformed `error_json` (invalid JSON syntax)
**Fix**: Add tests that directly corrupt the Call object fields returned by the mock to verify crashes, not coercion
**Priority**: P0

### Test: Hash collision handling (MISSING)
**Issue**: No test verifies behavior when multiple calls have same request_hash
**Evidence**: `find_call_by_request_hash` documentation (line 2654 in recorder.py) states "If multiple calls match (same request made twice), returns..." but behavior is not tested
**Fix**: Add test that records same request twice in source run, verify replayer returns the first/correct one
**Priority**: P1

### Test: Replayer with missing Call.call_id (MISSING)
**Issue**: No test verifies crash when Call object from audit trail has None/empty call_id
**Evidence**: ReplayPayloadMissingError uses `call.call_id` directly - if audit trail is corrupted and call_id is None, this would crash differently than expected
**Fix**: Add test with corrupted Call object (call_id=None) to verify appropriate crash, not AttributeError
**Priority**: P0

### Test: Response data type validation (MISSING)
**Issue**: No test verifies behavior when `get_call_response_data` returns non-dict type
**Evidence**: Replayer assumes `response_data` is dict but doesn't validate. If audit trail is corrupted and returns string/list, this could cause downstream errors
**Fix**: Add test where `get_call_response_data` returns `"not a dict"` and verify crash with clear message
**Priority**: P0

### Test: error_json parsing failure (MISSING)
**Issue**: No test verifies behavior when `call.error_json` contains invalid JSON
**Evidence**: Line 188 does `json.loads(call.error_json)` with no try/except - corrupted audit trail would crash with generic JSONDecodeError
**Fix**: Add test with malformed JSON in error_json field, verify crash with context (call_id, request_hash)
**Priority**: P1

### Test: Concurrent cache access (MISSING)
**Issue**: Documentation mentions thread safety (line 106) but no tests verify cache behavior under concurrent access
**Evidence**: "Thread Safety: The replayer caches results in memory. If used across threads, external synchronization may be needed for the cache."
**Fix**: Add test with ThreadPoolExecutor replaying same request simultaneously, verify both get correct response (no race condition)
**Priority**: P2

## Misclassified Tests

### Test: All tests in TestCallReplayer
**Issue**: These are unit tests with mocks, but should have integration test variants
**Evidence**: Tests use `MagicMock` for `LandscapeRecorder` but never test with real database. Replay correctness depends on actual SQL query correctness (JOIN conditions, WHERE clauses, etc.)
**Fix**: Add `tests/integration/test_replayer_database.py` with real LandscapeDB fixture that:
1. Records multiple calls with different request_hashes
2. Uses CallReplayer to replay them
3. Verifies correct calls are matched by hash
**Priority**: P1

### Test: test_replay_http_call_type (line 277)
**Issue**: Partially duplicates test_replay_returns_recorded_response with different call_type
**Evidence**: Same test logic, just different CallType enum. This could be a parameterized test.
**Fix**: Use `@pytest.mark.parametrize("call_type", ["llm", "http"])` to reduce duplication
**Priority**: P3

## Infrastructure Gaps

### Gap: No pytest fixture for pre-populated mock recorder
**Issue**: Every test method calls `_create_mock_recorder()` and repeats setup
**Evidence**: Lines 76-81 define helper method, but every test starts with `recorder = self._create_mock_recorder()`
**Fix**: Create `@pytest.fixture` for `mock_recorder` and inject it into test methods
**Priority**: P3

### Gap: No conftest.py with shared fixtures
**Issue**: No shared test fixtures for CallReplayer common scenarios
**Evidence**: Tests create ReplayedCall, Call, mock recorders repeatedly
**Fix**: Add `tests/plugins/clients/conftest.py` with:
- `@pytest.fixture` for `mock_recorder`
- `@pytest.fixture` for `sample_llm_request_data`
- `@pytest.fixture` for `sample_call_record` (factory fixture)
**Priority**: P3

### Gap: No property-based testing for replay determinism
**Issue**: Core property "same request hash = same response" is only tested with fixed examples
**Evidence**: Tests use hardcoded request_data dictionaries
**Fix**: Add Hypothesis property test:
```python
@given(st.dictionaries(st.text(), st.integers()))
def test_replay_determinism(request_data):
    # Record call with request_data
    # Replay twice
    # Assert responses identical
```
**Priority**: P2

### Gap: Missing edge case tests
**Issue**: No tests for:
- Empty request_data `{}`
- Very large request_data (e.g., 10MB JSON)
- Request with nested NaN/Infinity (should fail canonicalization)
- Unicode in request_data
- request_hash collision (different requests, same hash - astronomically unlikely but possible)
**Evidence**: Only "happy path" and error status tested
**Fix**: Add edge case test suite
**Priority**: P2

### Gap: No test for ReplayPayloadMissingError attributes
**Issue**: Test verifies exception is raised (line 271) but doesn't verify error message quality
**Evidence**: Line 274-275 check attributes exist but don't verify error message contains useful context for debugging
**Fix**: Add assertion: `assert "payload purged" in str(exc_info.value)` and verify call_id is in message
**Priority**: P3

## Audit Trail Integrity Gaps

### Gap: No test verifies response_data immutability
**Issue**: Cache returns response_data dict directly - mutation could affect future replays
**Evidence**: Line 164 returns `resp` from cache tuple, but if this is a mutable reference, caller could corrupt cache
**Fix**: Test that modifying `result.response_data` doesn't affect subsequent calls (cache should deepcopy or return frozen dict)
**Priority**: P2

### Gap: No test for payload store disabled scenario
**Issue**: No test verifies error message when payload store is explicitly disabled (vs. payload purged)
**Evidence**: `ReplayPayloadMissingError` message mentions "payload store may be disabled" but this scenario isn't tested
**Fix**: Add test that mocks payload store as None/disabled, verify error distinguishes this from purged payload
**Priority**: P3

### Gap: No test for malformed Call object structure
**Issue**: No test verifies behavior when Call object has unexpected None values for required fields
**Evidence**: Replayer accesses `call.status`, `call.error_json`, `call.latency_ms`, `call.call_id` without None checks
**Fix**: Add test with Call object where latency_ms=None, verify crash with context (Tier 1 trust violation)
**Priority**: P0

## Positive Observations

- Clean test structure with descriptive names following "what it does" pattern
- Good separation of concerns: ReplayedCall, exceptions, and CallReplayer tested independently
- Cache behavior is explicitly tested (test_replay_caches_results, clear_cache)
- Error cases are covered (ReplayMissError, error call status)
- Test helper methods (`_create_mock_recorder`, `_create_mock_call`) reduce duplication
- Tests verify both happy path and error scenarios
- Tests verify exact mock call parameters (line 135-139)
- Cache key isolation is tested (different call_types with same hash)
