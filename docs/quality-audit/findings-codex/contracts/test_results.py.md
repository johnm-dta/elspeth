# Test Defect Report

## Summary

- The "status is Literal, not enum" test only checks string equality and `isinstance(str)`, so a `StrEnum` value would still pass and the "not enum" contract isn't enforced.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/test_results.py:117` frames the "not enum" contract without a strict type check.
- `tests/contracts/test_results.py:127` uses `isinstance(..., str)`, which is true for `enum.StrEnum`.
- Code snippet:
```python
def test_status_is_literal_not_enum(self) -> None:
    ...
    assert success.status == "success"
    assert error.status == "error"
    assert isinstance(success.status, str)
    assert isinstance(error.status, str)
```

## Impact

- A regression to `StrEnum` (or another `str` subclass) would pass this test while violating the literal-string contract.
- Downstream code could start relying on enum behavior without the contract tests catching it.

## Root Cause Hypothesis

- The test equates "is str" with "not enum" and doesn't account for `str` subclasses.

## Recommended Fix

- Replace the `isinstance` checks with exact type checks, e.g., `assert type(success.status) is str` and `assert type(error.status) is str`.
- Optionally add a negative control using a `StrEnum` in-test to prove the assertion fails for enums.
---
# Test Defect Report

## Summary

- Tests use `hasattr` to check system-owned contract attributes, which is a defensive pattern prohibited by CLAUDE.md and does not enforce crash-on-anomaly behavior.

## Severity

- Severity: trivial
- Priority: P3

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/contracts/test_results.py:207` uses `hasattr` to assert absence of `AcceptResult`.
- `tests/contracts/test_results.py:296` and `tests/contracts/test_results.py:300` use `hasattr` to assert presence/absence of fields.
- Code snippet:
```python
assert not hasattr(results, "AcceptResult")
...
assert hasattr(descriptor, "artifact_type")
assert not hasattr(descriptor, "kind")
```

## Impact

- Normalizes defensive checks in tests, which conflicts with the "crash on anomaly" rule for system-owned contracts.
- Can allow missing-attribute regressions to slip if `__getattr__` or other fallbacks are introduced.

## Root Cause Hypothesis

- Convenience checks were used to avoid AttributeError instead of asserting strict attribute access.

## Recommended Fix

- Replace `hasattr` with direct attribute access for presence and `pytest.raises(AttributeError)` for absence.
- Example pattern: `with pytest.raises(AttributeError): _ = descriptor.kind` and `with pytest.raises(AttributeError): _ = results.AcceptResult`.
---
# Test Defect Report

## Summary

- ArtifactDescriptor "required field" and URL-validation tests only cover happy paths; they never assert the error paths that enforce required arguments or sanitized URLs.

## Severity

- Severity: minor
- Priority: P2

## Category

- Missing Edge Cases

## Evidence

- `tests/contracts/test_results.py:302` and `tests/contracts/test_results.py:314` describe required fields but only construct valid instances (no `pytest.raises` for missing args).
- `tests/contracts/test_results.py:375` and `tests/contracts/test_results.py:396` only pass sanitized URLs to factory methods.
- `src/elspeth/contracts/results.py:238` and `src/elspeth/contracts/results.py:269` define TypeError guards for unsanitized URLs that are untested.
- Code snippet:
```python
def test_content_hash_is_required(self) -> None:
    # This would fail at runtime with a TypeError if content_hash were omitted
    # We verify by constructing with all required fields
    descriptor = ArtifactDescriptor(...)
```

## Impact

- Regressions that make `content_hash`/`size_bytes` optional or bypass URL sanitization would not be caught.
- Weakens audit integrity guarantees and could allow unsanitized URLs into audit records.

## Root Cause Hypothesis

- The tests focus on positive construction and rely on comments rather than asserting error paths.

## Recommended Fix

- Add negative tests that omit required fields and expect `TypeError` from the dataclass constructor.
- Add factory-method tests that pass a plain string (or object without `sanitized_url`/`fingerprint`) to `for_database`/`for_webhook` and assert `TypeError`.
