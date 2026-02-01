# Test Defect Report

## Summary

- Property-based extra-fields test swallows exceptions, so crashes on unexpected fields are never caught

## Severity

- Severity: major
- Priority: P2

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/contracts/transform_contracts/test_transform_protocol.py:235` shows a broad try/except that ignores failures despite the docstring claiming “no crash”
```python
# tests/contracts/transform_contracts/test_transform_protocol.py
try:
    result = transform.process(input_with_extra, ctx)
    assert isinstance(result, TransformResult)
except Exception:
    # Some transforms may reject extra fields - that's valid behavior
    pass
```
- `src/elspeth/contracts/data.py:37` documents that extra fields are ignored by schema validation, so a crash here is a contract violation, not acceptable behavior
```python
# src/elspeth/contracts/data.py
# Features:
# - Extra fields ignored (rows may have more fields than schema requires)
model_config = ConfigDict(
    extra="ignore",
    strict=False,
    frozen=False,
)
```

## Impact

- Transforms can raise on extra fields (contrary to schema expectations) and the test will still pass
- This hides robustness regressions around common real-world inputs with extra keys
- False confidence that the transform contract tolerates schema-irrelevant fields

## Root Cause Hypothesis

- The test attempted to allow both success and error outcomes but used a blanket exception swallow to do so
- Confusion between “can return error” and “can raise exception”

## Recommended Fix

- Remove the try/except and require that `transform.process(...)` does not raise
- If extra fields are legitimately rejected, assert a structured `TransformResult.error` instead of permitting exceptions
- Example adjustment:
```python
result = transform.process(input_with_extra, ctx)
assert isinstance(result, TransformResult)
```
---
# Test Defect Report

## Summary

- Schema contract checks only verify `type`, not `PluginSchema` subclasses, so invalid schemas pass

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/transform_contracts/test_transform_protocol.py:87` only checks `type`, not the required schema base class
```python
# tests/contracts/transform_contracts/test_transform_protocol.py
assert hasattr(transform, "input_schema")
assert isinstance(transform.input_schema, type)
```
- `src/elspeth/plugins/protocols.py:145` defines the contract as `type["PluginSchema"]`
```python
# src/elspeth/plugins/protocols.py
input_schema: type["PluginSchema"]
output_schema: type["PluginSchema"]
```

## Impact

- A transform could set `input_schema = dict` (or any random class) and still pass contract tests
- Schema validation and pipeline compatibility checks can fail at runtime without test coverage
- Weakens guarantees about row validation and auditability metadata

## Root Cause Hypothesis

- Tests were written to keep assertions minimal and avoid importing PluginSchema
- Contract type expectations were not propagated into the test

## Recommended Fix

- Import `PluginSchema` and assert subclassing for both schemas
- Example:
```python
from elspeth.contracts import PluginSchema

assert issubclass(transform.input_schema, PluginSchema)
assert issubclass(transform.output_schema, PluginSchema)
```
---
# Test Defect Report

## Summary

- Error-contract tests never assert that `error_input` actually produces an error, so success results pass silently

## Severity

- Severity: major
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/transform_contracts/test_transform_protocol.py:270` states error_input should trigger an error, but all tests guard on `if result.status == "error"` without enforcing it
```python
# tests/contracts/transform_contracts/test_transform_protocol.py
result = transform.process(error_input, ctx)
if result.status == "error":
    assert result.reason is not None
```

## Impact

- A transform that incorrectly returns success for error cases will still pass all error-contract tests
- Error routing, retryability, and audit error recording are effectively untested
- Can mask serious failures until runtime

## Root Cause Hypothesis

- Base tests were written to be permissive for reuse, but the contract class is specifically for error behavior
- Missing explicit assertion that the fixture actually triggers error status

## Recommended Fix

- Assert `result.status == "error"` at the start of each error-contract test (or in a shared helper)
- If a transform cannot produce errors, it should not use `TransformErrorContractTestBase`
- Example:
```python
result = transform.process(error_input, ctx)
assert result.status == "error"
```
---
# Test Defect Report

## Summary

- Lifecycle hook tests skip execution when hooks are missing, allowing protocol violations to pass

## Severity

- Severity: minor
- Priority: P2

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/contracts/transform_contracts/test_transform_protocol.py:200` uses `hasattr` guards, so missing hooks don’t fail
```python
# tests/contracts/transform_contracts/test_transform_protocol.py
if hasattr(transform, "on_start"):
    transform.on_start(ctx)
```
- `src/elspeth/plugins/base.py:105` shows `BaseTransform` provides no-op `on_start`/`on_complete`, so direct calls are expected and safe
```python
# src/elspeth/plugins/base.py
def on_start(self, ctx: PluginContext) -> None:
    pass
def on_complete(self, ctx: PluginContext) -> None:
    pass
```

## Impact

- Transforms missing hooks (or with misspelled methods) will pass contract tests
- Engine lifecycle calls could raise `AttributeError` at runtime
- Contract “MUST not raise” is not actually enforced

## Root Cause Hypothesis

- Hooks treated as optional in tests despite being part of the protocol/base class
- Defensive guard added to avoid failures but weakens the contract

## Recommended Fix

- Remove the `hasattr` guards and call `transform.on_start(ctx)` / `transform.on_complete(ctx)` unconditionally
- If hooks are truly optional, explicitly assert their absence is acceptable in the contract documentation and tests
