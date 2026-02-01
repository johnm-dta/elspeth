# Test Defect Report

## Summary

- `test_no_retry_on_non_retryable` asserts only the exception type and does not verify the operation is called once, so unintended retries on non-retryable errors can slip through.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_retry.py:32-44` only asserts `TypeError` is raised, with no call count or retry callback tracking.
```python
def failing_operation() -> None:
    raise TypeError("Not retryable")

with pytest.raises(TypeError):
    manager.execute_with_retry(
        failing_operation,
        is_retryable=lambda e: isinstance(e, ValueError),
    )
```
- `src/elspeth/engine/retry.py:152-160` shows retry behavior is gated by `retry_if_exception(is_retryable)`, so the test should also verify the non-retryable path executes exactly once.

## Impact

- A regression that retries non-retryable failures but still re-raises `TypeError` would pass this test.
- Duplicate side effects (e.g., repeated writes or external calls) could go undetected.
- Creates false confidence that “no retry” semantics are enforced.

## Root Cause Hypothesis

- The test focuses on exception type and omits attempt-count verification.
- Similar exception-only assertions may exist elsewhere without validating call counts.

## Recommended Fix

- Add a call counter and assert it remains 1; optionally track `on_retry` and assert it is never called.
- Example:
```python
attempts: list[tuple[int, BaseException]] = []
call_count = 0

def failing_operation() -> None:
    nonlocal call_count
    call_count += 1
    raise TypeError("Not retryable")

with pytest.raises(TypeError):
    manager.execute_with_retry(
        failing_operation,
        is_retryable=lambda e: isinstance(e, ValueError),
        on_retry=lambda attempt, error: attempts.append((attempt, error)),
    )

assert call_count == 1
assert attempts == []
```
- Priority justification: this is core retry behavior; unintended retries are high-impact and should be directly asserted.
---
# Test Defect Report

## Summary

- `test_message_format` uses substring checks, so changes to the exception message format can pass unnoticed.

## Severity

- Severity: trivial
- Priority: P3

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_retry.py:189-196` only checks for substrings in the message.
```python
assert "5" in str(exc)
assert "original error" in str(exc)
```
- `src/elspeth/engine/retry.py:41-44` defines a specific message format: `Max retries ({attempts}) exceeded: {last_error}`.

## Impact

- Format regressions (missing prefix, punctuation, or ordering) would not be detected.
- Consumers relying on a stable message format (logs, error parsing) could be impacted without test coverage.

## Root Cause Hypothesis

- Substring matching was used to avoid brittleness, but the test name implies exact format validation.

## Recommended Fix

- Assert the full message or use a strict, anchored regex to match the exact format.
- Example:
```python
assert str(exc) == "Max retries (5) exceeded: original error"
```
- Priority justification: low-risk, quick tightening of a format-specific test.
