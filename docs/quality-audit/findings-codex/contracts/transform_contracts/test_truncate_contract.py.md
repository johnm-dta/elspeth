# Test Defect Report

## Summary

- Strict-mode contract setup defines an `error_input` but never asserts that it yields an error; the inherited base checks are conditional, so a regression that returns success would pass.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/transform_contracts/test_truncate_contract.py:63` and `tests/contracts/transform_contracts/test_truncate_contract.py:83` define strict-mode error fixtures but add no explicit assertion that `error_input` must return an error.
```python
class TestTruncateStrictContract(TransformErrorContractTestBase):
    ...
    @pytest.fixture
    def error_input(self) -> dict[str, Any]:
        return {"other_field": "value", "id": 2}
```
- `tests/contracts/transform_contracts/test_transform_protocol.py:282` and `tests/contracts/transform_contracts/test_transform_protocol.py:289` show the base error checks only run if `result.status == "error"`, so a success result would not fail the test.
```python
result = transform.process(error_input, ctx)
if result.status == "error":
    assert result.reason is not None
```
- `src/elspeth/plugins/transforms/truncate.py:112` shows strict mode is expected to return an error on missing field, so the negative path is a defined contract that should be asserted.
```python
if field_name not in output:
    if self._strict:
        return TransformResult.error(...)
```

## Impact

- Strict-mode regressions that incorrectly return success (or skip missing fields) would pass the contract suite, weakening confidence in error handling.
- Error routing/audit behavior tied to strict-mode failures could silently stop being exercised in this contract layer.

## Root Cause Hypothesis

- Over-reliance on the shared `TransformErrorContractTestBase`, which does not enforce that `error_input` must produce an error, and no plugin-specific negative assertion added here.

## Recommended Fix

- Add an explicit strict-mode negative test in `tests/contracts/transform_contracts/test_truncate_contract.py` to assert `status == "error"` and reason contents for `error_input`.
```python
def test_strict_missing_field_returns_error(
    self,
    transform: TransformProtocol,
    error_input: dict[str, Any],
    ctx: PluginContext,
) -> None:
    result = transform.process(error_input, ctx)
    assert result.status == "error"
    assert result.reason == {"reason": "missing_field", "field": "required_field"}
```
- This is a direct contract assertion for a defined error path and closes the gap without altering shared base behavior.
