# Test Defect Report

## Summary

- HTTP mocking is wired to `ctx.http_client`, but AzurePromptShield creates and uses its own internal `httpx.Client`, so the contract tests can hit real network calls and do not actually exercise the clean/attack responses they set up.

## Severity

- Severity: major
- Priority: P1

## Category

- [Misclassified Tests]

## Evidence

- `tests/contracts/transform_contracts/test_azure_prompt_shield_contract.py:44` configures `ctx.http_client.post` and the fixtures use that context (`tests/contracts/transform_contracts/test_azure_prompt_shield_contract.py:78`, `tests/contracts/transform_contracts/test_azure_prompt_shield_contract.py:110`), but the transform fixture never injects a mock client (`tests/contracts/transform_contracts/test_azure_prompt_shield_contract.py:61`).
- `src/elspeth/plugins/transforms/azure/prompt_shield.py:419` and `src/elspeth/plugins/transforms/azure/prompt_shield.py:462` show the transform calls `self._get_http_client()` and then `client.post(...)`, with no reference to `ctx.http_client`.

```python
# tests/contracts/transform_contracts/test_azure_prompt_shield_contract.py
def _make_mock_context(http_response: dict[str, Any]) -> Mock:
    ctx = Mock(spec=PluginContext)
    ...
    ctx.http_client.post.return_value = response_mock
    return ctx
```

```python
# src/elspeth/plugins/transforms/azure/prompt_shield.py
def _get_http_client(self) -> httpx.Client:
    if self._http_client is None:
        self._http_client = httpx.Client(timeout=30.0)
    return self._http_client

...
client = self._get_http_client()
response = client.post(...)
```

- Examples of what's missing: no patching of `AzurePromptShield._get_http_client` or injection of `transform._http_client` in the `transform` fixtures, so the prepared responses in `_make_clean_response()` and `_make_attack_response()` are never used.

## Impact

- Contract tests can become flaky or slow because they may attempt real HTTP calls to `https://test.cognitiveservices.azure.com` in CI.
- The tests do not validate behavior against the intended clean/attack responses, so regressions in response parsing or attack detection could slip through while the contract tests still pass.

## Root Cause Hypothesis

- The test assumes AzurePromptShield uses `PluginContext.http_client` (pattern from other plugins), but this transform uses an internal `httpx.Client`, so the mock is attached to the wrong object.
- Missing shared fixture pattern for HTTP-client injection in contract tests.

## Recommended Fix

- Inject a mock HTTP client directly into the transform in each `transform` fixture, or monkeypatch `_get_http_client` to return a stubbed client. Example pattern:
  ```python
  @pytest.fixture
  def transform(self) -> TransformProtocol:
      transform = AzurePromptShield({...})
      response = Mock()
      response.status_code = 200
      response.json.return_value = _make_clean_response()
      response.raise_for_status = Mock()
      mock_client = Mock()
      mock_client.post.return_value = response
      transform._http_client = mock_client
      return transform
  ```
- For the error contract class, use `_make_attack_response()` in the injected response so the error path is exercised.
- Priority justification: this removes external network dependence and makes the contract tests deterministic, preventing flaky CI failures and ensuring the intended behavior is actually exercised.
