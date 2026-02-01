# Test Quality Review: test_llm_transforms.py

## Summary

Integration tests for LLM transforms that verify end-to-end flows with mocked HTTP/SDK responses and real audit trail recording. Tests cover template rendering, API call recording, batch aggregation, error handling, and pooled execution. Found several critical architectural gaps around audit trail verification, missing negative test coverage for auditability contracts, and potential race conditions in pooled execution tests.

## Poorly Constructed Tests

### Test: test_template_rendering_api_call_response_parsing (line 103)
**Issue**: Incomplete auditability verification - does not verify request/response payloads are recorded
**Evidence**: Test verifies `calls[0].call_type`, `status`, and `latency_ms` but never checks `request_hash`, `response_hash`, or payload content. CLAUDE.md explicitly requires "Full request AND response recorded" (line 27).
**Fix**: Add assertions verifying:
```python
assert calls[0].request_hash is not None
assert calls[0].response_hash is not None
# Verify payloads are retrievable
request_payload = recorder.get_call_payload(calls[0].call_id, "request")
assert request_payload is not None
assert "messages" in request_payload
```
**Priority**: P0

### Test: test_audit_trail_records_template_hashes (line 167)
**Issue**: Only verifies hash field presence, not hash stability or correctness
**Evidence**: Tests that `llm_response_template_hash` exists and has length 64, but doesn't verify the hash is deterministic or correct for the given template.
**Fix**: Add test that renders same template twice and verifies hash stability:
```python
# Same template should produce same hash
result1 = transform.process({"text": "input"}, ctx)
result2 = transform.process({"text": "input"}, ctx)
assert result1.row["llm_response_template_hash"] == result2.row["llm_response_template_hash"]
assert result1.row["llm_response_variables_hash"] == result2.row["llm_response_variables_hash"]

# Different variables should produce different hash
result3 = transform.process({"text": "different"}, ctx)
assert result3.row["llm_response_template_hash"] == result1.row["llm_response_template_hash"]  # Same template
assert result3.row["llm_response_variables_hash"] != result1.row["llm_response_variables_hash"]  # Different vars
```
**Priority**: P1

### Test: test_api_error_recorded_in_audit_trail (line 214)
**Issue**: Incomplete error recording verification - only checks error exists, not error payload integrity
**Evidence**: Tests `calls[0].error_json is not None` and substring match, but doesn't verify error structure meets audit requirements or that request was still captured.
**Fix**: Verify complete error audit trail:
```python
assert calls[0].request_hash is not None  # Request captured even on error
assert calls[0].response_hash is None  # No response on error
error_data = json.loads(calls[0].error_json)
assert "error_type" in error_data
assert "error_message" in error_data
assert error_data["error_message"] == "API server error"
```
**Priority**: P1

### Test: test_system_prompt_included_when_configured (line 292)
**Issue**: No audit trail verification for system prompt inclusion
**Evidence**: Test only verifies the mock was called correctly, but doesn't verify the system prompt was recorded in the audit trail request payload.
**Fix**: Add audit trail verification:
```python
calls = recorder.get_calls(state_id)
request_payload = recorder.get_call_payload(calls[0].call_id, "request")
assert len(request_payload["messages"]) == 2
assert request_payload["messages"][0]["role"] == "system"
```
**Priority**: P2

### Test: test_http_client_call_and_response_parsing (line 378)
**Issue**: Missing audit trail verification entirely
**Evidence**: Test for OpenRouter HTTP integration but never verifies `recorder.get_calls()` or that HTTP request/response were recorded to audit trail.
**Fix**: Add complete audit trail verification (same pattern as line 159-165).
**Priority**: P0

### Test: test_http_error_returns_transform_error (line 429)
**Issue**: Doesn't verify error was recorded in audit trail
**Evidence**: Only verifies `TransformResult` error fields, ignores that OpenRouter should record HTTP errors to audit trail via `ctx.landscape` and `state_id`.
**Fix**: Add audit trail verification for HTTP errors.
**Priority**: P1

### Test: test_batch_submit_and_checkpoint_flow (line 533)
**Issue**: Accesses private `_checkpoint` attribute directly instead of using public API
**Evidence**: `checkpoint = ctx_with_checkpoint._checkpoint  # type: ignore[attr-defined]`
**Fix**: Use public checkpoint API:
```python
checkpoint = ctx_with_checkpoint.get_checkpoint()
assert checkpoint["batch_id"] == "batch-xyz789"
```
**Priority**: P2

### Test: test_batch_resume_and_completion_flow (line 570)
**Issue**: Directly mutates `ctx._checkpoint` instead of using `update_checkpoint()`
**Evidence**: Lines 577-590 use `ctx._checkpoint.update({...})` which violates encapsulation and bypasses checkpoint contract.
**Fix**: Use public API: `ctx.update_checkpoint({...})`
**Priority**: P2

### Test: test_batch_resume_and_completion_flow (line 570)
**Issue**: No audit trail verification for batch LLM calls
**Evidence**: Verifies results are returned but never checks if individual LLM calls from the batch were recorded to audit trail. Per CLAUDE.md line 27, "External calls - Full request AND response recorded" is non-negotiable.
**Fix**: Add verification that each batch row's LLM call was recorded:
```python
# For Azure Batch, verify that results include audit metadata
# OR verify batch file upload/download was recorded as external calls
```
**Priority**: P0

### Test: test_batch_with_simulated_capacity_errors (line 1017)
**Issue**: Uses mock process function that bypasses audit trail entirely
**Evidence**: Comment admits "uses mock process function because capacity retries would cause call_index collisions" - this is a bug in the implementation being tested, not a reason to skip audit verification.
**Fix**: Fix the underlying call_index collision bug in `AuditedHTTPClient`, then test with real audit trail recording.
**Priority**: P0

### Test: test_batch_capacity_retry_timeout (line 1096)
**Issue**: Same as above - uses mock to bypass audit trail recording
**Evidence**: Mock process function means zero audit trail verification for capacity timeout scenario.
**Fix**: Fix call_index collision bug and use real audit trail.
**Priority**: P0

### Test: test_batch_mixed_results (line 1141)
**Issue**: Creates file-based SQLite for "thread safety" but never verifies thread safety
**Evidence**: Comment says "Use file-based SQLite for thread safety" (line 1191) but test doesn't actually verify concurrent access works correctly or that no races occur.
**Fix**: Either remove the misleading comment (if thread safety isn't being tested) or add explicit concurrent verification.
**Priority**: P2

### Test: test_pool_size_1_uses_sequential_processing (line 916)
**Issue**: Missing audit trail verification
**Evidence**: Verifies result status but not that the call was recorded to audit trail.
**Fix**: Add `recorder.get_calls(state.state_id)` verification.
**Priority**: P1

## Infrastructure Gaps

### Gap: No fixture for creating valid state with audit trail
**Issue**: Every test manually constructs `run -> node -> row -> token -> state` (60+ lines of duplication across classes)
**Evidence**: Lines 72-101, 347-376, 781-806 all repeat identical setup code.
**Fix**: Create shared fixture:
```python
@pytest.fixture
def audit_state(recorder: LandscapeRecorder) -> tuple[str, str, str]:
    """Create run, node, state for testing.

    Returns: (run_id, node_id, state_id)
    """
    schema = SchemaConfig.from_dict({"fields": "dynamic"})
    run = recorder.begin_run(config={}, canonical_version="v1")
    node = recorder.register_node(...)
    row = recorder.create_row(...)
    token = recorder.create_token(row_id=row.row_id)
    state = recorder.begin_node_state(...)
    return run.run_id, node.node_id, state.state_id
```
**Priority**: P1

### Gap: No shared test for audit trail completeness contract
**Issue**: Every LLM transform test should verify the same auditability contract, but verification is scattered and incomplete.
**Evidence**: Some tests check `request_hash`, some check `error_json`, none check payload retrieval comprehensively.
**Fix**: Create reusable audit trail verification helper:
```python
def assert_llm_call_audit_complete(
    recorder: LandscapeRecorder,
    state_id: str,
    expected_status: CallStatus,
    verify_request: bool = True,
    verify_response: bool = True,
) -> None:
    """Verify LLM call meets auditability contract."""
    calls = recorder.get_calls(state_id)
    assert len(calls) == 1
    call = calls[0]
    assert call.call_type == CallType.LLM
    assert call.status == expected_status
    assert call.latency_ms is not None

    if verify_request:
        assert call.request_hash is not None
        payload = recorder.get_call_payload(call.call_id, "request")
        assert payload is not None

    if verify_response:
        if expected_status == CallStatus.SUCCESS:
            assert call.response_hash is not None
            payload = recorder.get_call_payload(call.call_id, "response")
            assert payload is not None
        else:
            assert call.response_hash is None
            assert call.error_json is not None
```
**Priority**: P0

### Gap: No test coverage for concurrent audit trail writes in pooled execution
**Issue**: Pooled execution has multiple threads writing to audit trail simultaneously, but no test verifies thread safety.
**Evidence**: `TestPooledExecutionIntegration` tests pooled execution but comments admit audit trail testing is problematic (line 1008-1014).
**Fix**: Create explicit concurrent write test using file-based SQLite and verify no corruption, missing calls, or race conditions.
**Priority**: P0

### Gap: No test for audit trail payload retention policy
**Issue**: CLAUDE.md mentions "Payload Store - Separates large blobs from audit tables with retention policies" but no test verifies LLM payloads respect retention.
**Evidence**: Tests create payloads but never verify they can be retrieved, deleted per policy, or that hashes survive deletion.
**Fix**: Add test:
```python
def test_llm_payload_retention_policy(recorder: LandscapeRecorder):
    # Record call with payload
    # Verify payload retrievable
    # Simulate payload deletion (retention policy trigger)
    # Verify hash still exists and matches
    # Verify payload retrieval returns None after deletion
```
**Priority**: P1

### Gap: No test for template rendering errors
**Issue**: Tests verify API errors but not template rendering failures (missing variables, syntax errors).
**Evidence**: Line 684 has comment about "missing required template field" but test doesn't verify what happens when template render fails.
**Fix**: Add explicit test:
```python
def test_template_render_error_quarantines_row():
    transform = ConcreteLLMTransform({
        "template": "{{ row.missing_field }}",
        "schema": DYNAMIC_SCHEMA,
    })
    result = transform.process({"other_field": "x"}, ctx)
    assert result.status == "error"
    assert result.reason["reason"] == "template_render_failed"
```
**Priority**: P2

### Gap: No test for NaN/Infinity in LLM responses
**Issue**: CLAUDE.md line 324 explicitly requires rejecting NaN/Infinity in canonical JSON, but no test verifies LLM responses with NaN are handled correctly.
**Evidence**: Mock responses use hardcoded strings, never problematic float values.
**Fix**: Add test where LLM response contains `{"score": NaN}` and verify it's rejected/quarantined.
**Priority**: P1

### Gap: Missing test for rate limit header parsing
**Issue**: Tests mock 429 errors but don't verify parsing of `Retry-After` headers or backoff metadata recording.
**Evidence**: Line 260 raises generic "Error 429" exception, doesn't verify `Retry-After: 60` is captured in audit trail.
**Fix**: Add test with realistic 429 response including headers, verify backoff metadata recorded.
**Priority**: P2

### Gap: No test verifying `on_start()` and `close()` lifecycle
**Issue**: Only one test calls `transform.on_start()` (line 1212), only one calls `transform.close()` (line 1002). Lifecycle is critical for pooled executors.
**Evidence**: Most tests don't initialize/cleanup properly, relying on Python GC instead of explicit lifecycle.
**Fix**: Add fixture that ensures proper lifecycle:
```python
@pytest.fixture
def transform_with_lifecycle(transform, ctx):
    transform.on_start(ctx)
    yield transform
    transform.close()
```
**Priority**: P2

## Misclassified Tests

### Test: All tests in TestAzureBatchLLMTransformIntegration (line 508)
**Issue**: Tests use mocks for all Azure SDK interactions - these are unit tests, not integration tests
**Evidence**: Lines 536-544, 593-638, 669-678 all use `Mock()` for Azure client. No real Azure SDK code is executed.
**Fix**: Either:
1. Move to `tests/unit/plugins/llm/test_azure_batch.py` and rename as unit tests
2. OR add real integration tests using Azure SDK test doubles (not unittest.Mock)
**Priority**: P1

### Test: test_batch_with_simulated_capacity_errors (line 1017)
**Issue**: Uses mock process function - this is a unit test of the pooling infrastructure, not an integration test
**Evidence**: Line 1048 defines `mock_process()` that doesn't call any real code paths.
**Fix**: Move to `tests/unit/plugins/test_pooling.py` or fix to use real process function.
**Priority**: P1

### Test: test_batch_capacity_retry_timeout (line 1096)
**Issue**: Same as above - mock process function makes this a unit test
**Evidence**: Line 1117 defines `mock_process_always_fails()`.
**Fix**: Move to unit tests or rewrite with real integration.
**Priority**: P1

## Missing Test Coverage

### Missing: Verify LLM response exceeds max tokens
**Scenario**: LLM response is truncated due to max_tokens limit
**Expected**: Audit trail records truncation, result includes truncation flag
**Priority**: P2

### Missing: Verify concurrent template rendering is thread-safe
**Scenario**: Pooled execution renders same template in multiple threads
**Expected**: No race conditions, all renders produce correct output
**Priority**: P1

### Missing: Verify batch partial failure cleanup
**Scenario**: Azure Batch API call succeeds but only 3/5 rows return results (2 missing from output file)
**Expected**: System detects missing rows, returns error results for them
**Priority**: P1

### Missing: Verify OpenRouter fallback model selection
**Scenario**: Primary model returns 404, OpenRouter auto-routes to fallback
**Expected**: Audit trail records which model actually processed the request
**Priority**: P2

### Missing: Verify secrets are not stored in audit trail
**Scenario**: API key is in config, LLM call is recorded
**Expected**: `request_hash` exists, payload contains HMAC fingerprint of API key, not plaintext (per CLAUDE.md line 360)
**Priority**: P0

### Missing: Verify deterministic hashing for identical requests
**Scenario**: Same template + variables produces same request hash
**Expected**: Two identical calls have same `request_hash`, enabling deduplication
**Priority**: P1

### Missing: Verify malformed JSON in LLM response
**Scenario**: LLM returns invalid JSON when JSON mode enabled
**Expected**: Error result with `reason: "json_parse_failed"`, raw response in audit trail
**Priority**: P2

### Missing: Verify batch resume with stale checkpoint
**Scenario**: Checkpoint contains `batch_id` but batch was already downloaded/deleted from Azure
**Expected**: System detects stale checkpoint, returns error (not silent corruption)
**Priority**: P1

## Positive Observations

- Tests correctly use in-memory databases for fast execution
- Good separation between `BaseLLMTransform`, `OpenRouterLLMTransform`, and `AzureBatchLLMTransform` test classes
- Concrete test helper class `ConcreteLLMTransform` avoids abstract class instantiation issues
- Test names are descriptive and follow naming convention
- Proper use of `pytest.raises()` for expected exceptions
- Mixed result testing (line 1141) shows good edge case coverage for pooled execution
