# Test Defect Report

## Summary

- Contract tests for AzureMultiQueryLLMTransform do not verify audit trail recording for LLM calls, so audit regressions can pass unnoticed.

## Severity

- Severity: major
- Priority: P1

## Category

- [Missing Audit Trail Verification]

## Evidence

- `tests/contracts/transform_contracts/test_azure_multi_query_contract.py:35` shows the context uses a loose Mock landscape and only stubs `record_external_call`, with no later assertions on audit recording:

```python
def _make_mock_context() -> Mock:
    ctx = Mock(spec=PluginContext)
    ctx.run_id = "test-run-001"
    ctx.state_id = "state-001"
    ctx.landscape = Mock()
    ctx.landscape.record_external_call = Mock()
    return ctx
```

- `tests/contracts/transform_contracts/test_azure_multi_query_contract.py:101` demonstrates the file’s explicit tests only check `_query_specs` and flags; there are no assertions on `ctx.landscape` or audit tables.
- `src/elspeth/plugins/llm/azure_multi_query.py:148` constructs an `AuditedLLMClient` tied to the recorder, and `src/elspeth/plugins/clients/llm.py:181` records each call via `record_call`, which the contract tests never verify:

```python
self._recorder.record_call(
    state_id=self._state_id,
    call_index=call_index,
    call_type=CallType.LLM,
    status=CallStatus.SUCCESS,
    request_data=request_data,
    response_data={...},
    latency_ms=latency_ms,
)
```

## Impact

- Auditability regressions (missing or malformed call records, wrong call_index ordering, missing request/response payloads) can ship undetected.
- Violates ELSPETH’s auditability standard and undermines traceability for LLM decisions.
- Creates false confidence that external calls are being properly recorded when only functional outputs are checked.

## Root Cause Hypothesis

- Contract tests are scoped narrowly to TransformProtocol behavior and rely on mocked context, so audit trail requirements were omitted from assertions.

## Recommended Fix

- Add audit-focused tests in `tests/contracts/transform_contracts/test_azure_multi_query_contract.py` that run `process()` and assert `ctx.landscape.record_call` is invoked once per query with `CallType.LLM`, `CallStatus.SUCCESS`, correct `state_id`, and expected request/response payloads (model, messages, usage).
- Add an error-path test (force `chat.completions.create` to raise) and assert `record_call` is invoked with `CallStatus.ERROR` and a retryable classification.
- If available, use a real `LandscapeRecorder` fixture and assert `node_states`, `token_outcomes`, and `artifacts` rows are written for this transform to meet the Auditability Standard.
- Priority justification: Audit trail integrity is a core contract; missing verification could allow silent audit data loss in production.
