Using skill: using-quality-engineering (test-maintenance-patterns) to guide the test-quality audit.

# Test Defect Report

## Summary

- Uses `hasattr` to validate logger interface, which is a prohibited defensive pattern and doesn't verify actual behavior.

## Severity

- Severity: minor
- Priority: P3

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/core/test_logging.py:22-25` uses `hasattr` checks on system logger methods.
```python
logger = get_logger("test")
assert hasattr(logger, "info")
assert hasattr(logger, "error")
assert hasattr(logger, "bind")
```
- No callability or behavior validation for `info`/`error`/`bind` in this test; a non-callable attribute would still pass.
- Example of what's missing: a direct call with output assertion to prove the logger actually logs.

## Impact

- Test passes even if `get_logger` returns an object with stub attributes; interface regressions can slip.
- Violates the repo’s no-defensive-patterns rule, creating false confidence in API shape.

## Root Cause Hypothesis

- Placeholder smoke test focused on attribute presence instead of behavior; defensive-check habit.

## Recommended Fix

- Replace `hasattr` checks with behavior assertions: call `logger.info`/`logger.error`, capture output, and assert expected fields.
- Optionally remove this test if behavior is already covered elsewhere to avoid duplication.
- Priority: P3 because it is a weak guard that can allow regressions in the core logging API.
---
# Test Defect Report

## Summary

- `test_logger_binds_context` does not assert that bound context is emitted; only checks instance identity.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/test_logging.py:63-72` only asserts non-None and identity.
```python
logger = get_logger("test")
bound = logger.bind(run_id="abc123")

assert bound is not None
assert bound is not logger
```
- No assertion that `run_id` appears in structured output or context.

## Impact

- A regression where `bind` drops context would go undetected; logs lose critical run identifiers.
- Test claims to verify context binding but does not validate the observable behavior.

## Root Cause Hypothesis

- Minimal check added to confirm API surface without validating end-to-end logging output.

## Recommended Fix

- Configure JSON logging in this test, log with `bound`, capture output, and assert `run_id` in parsed JSON (e.g., `data["run_id"] == "abc123"`).
- Use `capsys` (already used in this file) to keep assertions black-box and avoid private attribute access.
- Priority: P2 because it covers correctness of logging context propagation, which is essential for traceability.
