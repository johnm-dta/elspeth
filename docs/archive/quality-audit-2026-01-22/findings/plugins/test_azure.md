# Test Quality Review: test_azure.py

## Summary
The test suite has reasonable coverage of basic functionality but suffers from critical gaps in audit trail verification, incomplete verification of external call recording, inadequate testing of error scenarios, and infrastructure duplication. Tests verify response enrichment but fail to verify the auditability guarantees that are fundamental to ELSPETH's architecture.

## Poorly Constructed Tests

### Test: test_successful_llm_call_returns_enriched_row (line 345)
**Issue**: Incomplete verification - tests response enrichment but NOT audit trail recording
**Evidence**:
```python
# Tests response fields but never verifies record_call was invoked correctly
assert result.row["llm_response"] == "The analysis is positive."
assert result.row["llm_response_usage"] == {"prompt_tokens": 10, "completion_tokens": 25}
# ... but no verification of ctx.landscape.record_call()
```
**Fix**: Add assertions to verify `ctx.landscape.record_call` was called with:
- Correct `call_type=CallType.LLM`
- Correct `status=CallStatus.SUCCESS`
- Full request_data (model, messages, temperature, provider)
- Full response_data (content, model, usage)
- Non-zero latency_ms

**Priority**: P0 - Violates auditability standard "Full request AND response recorded"

### Test: test_calls_are_recorded_to_landscape (line 598)
**Issue**: Sleepy assertion - only verifies `record_call.called`, not correctness
**Evidence**:
```python
# Verify record_call was called (by AuditedLLMClient)
assert ctx.landscape is not None
assert ctx.landscape.record_call.called  # type: ignore[attr-defined]
```
**Fix**: Verify complete call signature:
```python
call_args = ctx.landscape.record_call.call_args
assert call_args.kwargs["call_type"] == CallType.LLM
assert call_args.kwargs["status"] == CallStatus.SUCCESS
assert "messages" in call_args.kwargs["request_data"]
assert "content" in call_args.kwargs["response_data"]
assert call_args.kwargs["latency_ms"] > 0
```
**Priority**: P0 - Current test provides zero confidence in audit recording

### Test: test_llm_client_error_returns_transform_error (line 386)
**Issue**: Missing audit trail verification for error recording
**Evidence**: Tests return value but doesn't verify the error was recorded to Landscape
```python
assert result.status == "error"
assert result.reason["reason"] == "llm_call_failed"
# Missing: Verify record_call was invoked with status=CallStatus.ERROR
```
**Fix**: Add verification that `record_call` was called with:
- `status=CallStatus.ERROR`
- `error={"type": "...", "message": "API Error", "retryable": False}`
- Full request_data preserved

**Priority**: P0 - Auditability requirement: "Every decision must be traceable"

### Test: test_rate_limit_error_is_retryable (line 398)
**Issue**: Missing audit trail verification for rate limit errors
**Evidence**: Same as above - doesn't verify error recording
**Fix**: Verify `record_call` was invoked with:
- `status=CallStatus.ERROR`
- `error={"retryable": True, ...}`
**Priority**: P0

### Test: test_template_rendering_error_returns_transform_error (line 375)
**Issue**: Incomplete error scenario - doesn't verify template errors are NOT recorded as LLM calls
**Evidence**: Template errors happen before LLM call, so `record_call` should NOT be invoked
**Fix**: Add assertion: `assert not ctx.landscape.record_call.called`
**Priority**: P1 - Prevents false audit records

### Test: test_batch_with_partial_failures (line 890)
**Issue**: No verification of per-row audit recording in batch mode
**Evidence**: Tests response fields but not that each successful row got a separate `record_call` with unique `call_index`
**Fix**: Verify `record_call.call_count == 3` (one per row), and verify call_index values are 0, 1, 2
**Priority**: P0 - Batch audit trail must be verifiable per-row

### Test: test_process_single_with_state_returns_error_on_llm_client_error (line 756)
**Issue**: Test patches too deep - patches `AuditedLLMClient.chat_completion` instead of letting real code run
**Evidence**:
```python
with patch("elspeth.plugins.clients.llm.AuditedLLMClient.chat_completion", side_effect=LLMClientError(...)):
```
**Fix**: Patch the underlying Azure client to raise exception, let AuditedLLMClient record it naturally, then verify the recording happened correctly
**Priority**: P1 - Current test doesn't verify the real error recording path

### Test: test_azure_client_created_with_correct_credentials (line 499)
**Issue**: Redundant verification - credentials are already tested in integration test
**Evidence**: This is a unit test verifying implementation details (how AzureOpenAI is instantiated), which duplicates integration test coverage
**Fix**: Delete this test - credential passing is sufficiently verified by successful LLM calls in integration tests
**Priority**: P3 - Low value, increases maintenance burden

### Test: test_close_is_noop (line 495)
**Issue**: Wrong assertion - close() is NOT a noop when pool_size > 1
**Evidence**: Line 592-597 in azure.py shows close() shuts down executor, clears recorder, clears clients
**Fix**: Test both cases:
- `pool_size=1`: close() doesn't crash (no executor)
- `pool_size=3`: close() shuts down executor and clears state
**Priority**: P2 - Test name and behavior mismatch

### Test: test_close_shuts_down_executor (line 683)
**Issue**: Incomplete verification - doesn't verify executor.shutdown was called
**Evidence**: Comment says "can't easily verify, but shouldn't raise" - this is lazy testing
**Fix**: Use `patch` or `Mock` on executor to verify `shutdown(wait=True)` was called
**Priority**: P2

## Misclassified Tests

### Test Class: TestAzureLLMTransformIntegration (line 529)
**Issue**: Not integration tests - these are unit tests with mocks
**Evidence**: All tests in this class use `mock_azure_openai_client()`, making them unit tests
**Fix**:
- Rename to `TestAzureLLMTransformEdgeCases` or merge into `TestAzureLLMTransformProcess`
- Create TRUE integration tests that hit a test OpenAI endpoint (with explicit opt-in flag)
**Priority**: P3 - Misleading classification, but tests are still useful

## Infrastructure Gaps

### Gap: Audit Trail Verification Fixture
**Issue**: Every test that calls `process()` should verify audit recording, but verification is manual and incomplete
**Evidence**: Only one test (`test_calls_are_recorded_to_landscape`) attempts verification, and it's incomplete
**Fix**: Create fixture that provides a verifier:
```python
@pytest.fixture
def audit_verifier(mock_recorder):
    """Helper to verify audit trail recordings."""
    def verify_llm_call_recorded(call_type, status, has_request=True, has_response=True):
        assert mock_recorder.record_call.called
        call_args = mock_recorder.record_call.call_args
        assert call_args.kwargs["call_type"] == call_type
        assert call_args.kwargs["status"] == status
        if has_request:
            assert "request_data" in call_args.kwargs
        if has_response and status == CallStatus.SUCCESS:
            assert "response_data" in call_args.kwargs
        return call_args  # For further assertions
    return verify_llm_call_recorded
```
**Priority**: P0 - Enables systematic audit verification

### Gap: Missing Secret Sanitization Tests
**Issue**: CLAUDE.md mandates HMAC fingerprints for secrets, but no tests verify API key is NOT stored in audit trail
**Evidence**: Secret handling section line 358-363 in CLAUDE.md, but no corresponding test
**Fix**: Add test that verifies `api_key` is NOT in request_data recorded to Landscape:
```python
def test_api_key_not_recorded_in_audit_trail(ctx, transform):
    with mock_azure_openai_client():
        transform.process({"text": "hello"}, ctx)

    call_args = ctx.landscape.record_call.call_args
    request_data = call_args.kwargs["request_data"]
    assert "api_key" not in str(request_data)  # Recursive check
```
**Priority**: P0 - Security/audit integrity requirement

### Gap: Missing Hash Determinism Tests
**Issue**: Response includes template_hash and variables_hash, but no tests verify these are deterministic
**Evidence**: Lines 361-362 in test verify hashes exist, but not that same inputs produce same hashes
**Fix**: Add test:
```python
def test_template_hashes_are_deterministic(ctx, transform):
    with mock_azure_openai_client():
        result1 = transform.process({"text": "hello"}, ctx)
        result2 = transform.process({"text": "hello"}, ctx)

    assert result1.row["llm_response_template_hash"] == result2.row["llm_response_template_hash"]
    assert result1.row["llm_response_variables_hash"] == result2.row["llm_response_variables_hash"]
```
**Priority**: P1 - Canonicalization is critical for audit integrity

### Gap: No Tests for Lookup/Template Source Fields
**Issue**: Lines 277-278, 515-516 in azure.py show template_source and lookup_source/hash are recorded, but no tests verify this
**Evidence**: Tests don't exercise `template_source`, `lookup_source`, or `lookup` config
**Fix**: Add tests with:
```python
"template": "{{ row.text }}",
"template_source": "templates/analysis.jinja2",
"lookup": {"category_map": {...}},
"lookup_source": "lookups/categories.yaml"
```
Verify these appear in response fields
**Priority**: P2 - Audit trail completeness

### Gap: No Tests for system_prompt_source
**Issue**: Line 279, 517 record `system_prompt_source`, but no tests verify it
**Evidence**: test_system_prompt_included_in_messages (line 424) tests system_prompt but not system_prompt_source
**Fix**: Add test:
```python
"system_prompt": "You are helpful.",
"system_prompt_source": "prompts/system.txt"
# Verify system_prompt_source in output
```
**Priority**: P2

### Gap: Repeated Mock Setup
**Issue**: `mock_azure_openai_client` context manager duplicated in every test
**Evidence**: Lines 347-351, 369, 378, etc. - same pattern repeated
**Fix**: Consider autouse fixture for common case:
```python
@pytest.fixture
def auto_mock_azure(mock_azure_openai_client):
    """Automatically mock Azure client with default response."""
    with mock_azure_openai_client():
        yield
```
Tests can override when needed
**Priority**: P3 - Quality of life improvement

### Gap: No Tests for Client Caching Logic
**Issue**: Lines 143-148 show complex LLM client caching with thread locks, but no tests verify:
- Clients are reused within a batch
- Clients are evicted after batch completes (line 425)
- Thread safety of cache access
**Evidence**: No test verifies `_llm_clients` cache behavior
**Fix**: Add tests:
- Verify same state_id reuses client (call_index increments)
- Verify cache eviction after batch
- Stress test with concurrent access (if pooled execution is tested)
**Priority**: P1 - Cache bugs would break call_index uniqueness

### Gap: Missing call_index Verification
**Issue**: call_index is critical for audit trail uniqueness (state_id, call_index) but no tests verify monotonic increment
**Evidence**: Batch tests don't verify call_index=0,1,2 for rows 0,1,2
**Fix**: Add to batch tests:
```python
calls = [call for call in ctx.landscape.record_call.call_args_list]
assert calls[0].kwargs["call_index"] == 0
assert calls[1].kwargs["call_index"] == 1
assert calls[2].kwargs["call_index"] == 2
```
**Priority**: P0 - Audit trail uniqueness requirement

### Gap: No Tests for on_error Config
**Issue**: Line 122 stores `_on_error` config, but no tests exercise it
**Evidence**: No test with `on_error: "quarantine"` or `on_error: "fail"`
**Fix**: Add tests verifying on_error behavior (if implemented in process())
**Priority**: P2 - Config coverage

### Gap: Missing Latency Verification
**Issue**: Audit trail records latency_ms, but no tests verify it's non-zero and reasonable
**Evidence**: No assertion on latency in any test
**Fix**: Add to successful call tests:
```python
call_args = ctx.landscape.record_call.call_args
assert call_args.kwargs["latency_ms"] > 0
assert call_args.kwargs["latency_ms"] < 60000  # Sanity check: under 1 minute
```
**Priority**: P2 - Latency is audit metadata

## Missing Test Scenarios

### Scenario: Empty String Response
**Issue**: What if LLM returns empty string? Is this recorded correctly?
**Evidence**: No test with `content=""`
**Fix**: Add test verifying empty response is recorded and doesn't crash
**Priority**: P2

### Scenario: Very Long Response
**Issue**: What if LLM returns 10KB of text? Does audit recording handle it?
**Evidence**: No test with large response
**Fix**: Add test with multi-kilobyte response
**Priority**: P3 - Regression prevention

### Scenario: Unicode and Special Characters
**Issue**: Canonical JSON must handle unicode correctly
**Evidence**: No test with emoji, CJK characters, or control characters in prompt/response
**Fix**: Add test with `{"text": "Hello 👋 世界 \n\t test"}`
**Priority**: P2 - Canonicalization correctness

### Scenario: Multiple Calls in Same State
**Issue**: If transform is called twice on same state_id (e.g., retry), call_index should increment
**Evidence**: No test verifies this
**Fix**: Add test calling `process()` twice with same ctx, verify call_index=0 then 1
**Priority**: P0 - Audit uniqueness

### Scenario: Concurrent Batch Processing
**Issue**: Batch processing claims thread safety (line 146), but no concurrency tests
**Evidence**: No test with pool_size > 1 actually executing concurrently
**Fix**: Add test with pool_size=3 and mock that verifies concurrent execution (check _llm_clients has correct state during execution)
**Priority**: P1 - Thread safety claims must be tested

### Scenario: temperature=1.0 (Non-Deterministic)
**Issue**: Tests only use default temperature=0.0, doesn't verify high-temperature recording
**Evidence**: test_temperature_and_max_tokens_passed_to_client uses temperature=0.7, but doesn't verify it's in audit trail
**Fix**: Add assertion verifying temperature is in request_data
**Priority**: P2

### Scenario: Provider Field Verification
**Issue**: AzureOpenAI sets provider="azure", but no test verifies this appears in audit trail
**Evidence**: No test checks request_data["provider"] == "azure"
**Fix**: Add assertion in audit verification tests
**Priority**: P1 - Provider is audit metadata

## Positive Observations

- Good use of fixtures to reduce setup duplication (`mock_recorder`, `ctx`, `transform`)
- Comprehensive config validation tests (lines 70-201)
- Good test naming and docstrings
- Appropriate use of context manager for mocking (`mock_azure_openai_client`)
- Tests cover both sequential and pooled execution modes
- Batch processing edge cases are tested (empty batch, all failures, partial failures)

## Recommendations

1. **Immediate (P0)**:
   - Add audit trail verification to every test that calls `process()`
   - Create audit_verifier fixture
   - Add secret sanitization test
   - Verify call_index monotonicity in batch tests
   - Fix incomplete audit verification in error tests

2. **High Priority (P1)**:
   - Add hash determinism tests
   - Test client cache eviction behavior
   - Add concurrent execution test for thread safety claims
   - Test template_source and lookup fields
   - Fix deep patching in error tests

3. **Medium Priority (P2)**:
   - Add latency verification
   - Test unicode/special characters
   - Test empty responses
   - Test on_error config
   - Add system_prompt_source test

4. **Low Priority (P3)**:
   - Rename "Integration" test class
   - Consider autouse mock fixture
   - Delete redundant credential test
   - Add very long response test

## Architecture Concerns

The test suite focuses heavily on "does it work?" and insufficiently on "can we prove what happened?". For an auditability-first framework, every test should verify the audit trail is complete and correct. The current suite would allow a regression where LLM calls succeed but aren't recorded - which violates ELSPETH's core guarantee.

**Suggested Standard**: Every test that exercises `process()` must include a corresponding `verify_audit_trail()` assertion as a mandatory step.
