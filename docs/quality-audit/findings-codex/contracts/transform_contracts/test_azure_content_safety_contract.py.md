Using skill: using-quality-engineering (contract-testing) to assess contract-test isolation and external dependency risks.

# Test Defect Report

## Summary

- Contract tests attempt to mock `PluginContext.http_client`, but AzureContentSafety uses its own internal `httpx.Client`, so the mock is ignored and tests can hit real HTTP endpoints.

## Severity

- Severity: major
- Priority: P1

## Category

- Misclassified Tests

## Evidence

- `tests/contracts/transform_contracts/test_azure_content_safety_contract.py:4` claims HTTP is mocked and `_make_mock_context` wires `ctx.http_client.post`, but this mock is never connected to the transform.
```python
"""... These tests mock the HTTP client ..."""
...
ctx.http_client.post.return_value = response_mock
```
- `src/elspeth/plugins/transforms/azure/content_safety.py:478` shows the transform creates and uses its own client via `_get_http_client()` and calls `client.post(...)`, bypassing `ctx.http_client`.
```python
client = self._get_http_client()
...
response = client.post(
    url,
    json=request_data,
    headers={
        "Ocp-Apim-Subscription-Key": self._api_key,
        "Content-Type": "application/json",
    },
)
```

## Impact

- Contract tests can make real network calls, causing flakiness or failures in CI without network access.
- Tests may pass while exercising only network-error paths, so safe/violation behavior is not validated.
- False confidence: regressions in response handling can slip through because mocked responses are never used.

## Root Cause Hypothesis

- The test setup assumes the transform uses `PluginContext.http_client`, but the implementation was refactored to use an internal `httpx.Client`, and the contract tests were not updated.

## Recommended Fix

- Inject a mock HTTP client into the transform (or monkeypatch `_get_http_client`) in `tests/contracts/transform_contracts/test_azure_content_safety_contract.py` instead of stubbing `ctx.http_client`.
```python
@pytest.fixture
def ctx(self, transform) -> PluginContext:
    response = Mock()
    response.status_code = 200
    response.json.return_value = _make_safe_response()
    response.raise_for_status = Mock()

    mock_client = Mock()
    mock_client.post.return_value = response
    transform._http_client = mock_client

    return PluginContext(
        run_id="test-run-001",
        config={},
        node_id="test",
        plugin_name="azure_content_safety",
    )
```
- Priority justification: removes unintended external HTTP calls and ensures contract tests exercise deterministic, mocked responses.
