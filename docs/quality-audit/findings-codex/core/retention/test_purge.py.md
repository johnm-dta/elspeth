# Test Defect Report

## Summary

- `purge_payloads` failure handling (`failed_refs`) is untested because the mock store cannot simulate delete failures for existing refs, so that branch is never exercised

## Severity

- Severity: minor
- Priority: P2

## Category

- Incomplete Contract Coverage

## Evidence

- `tests/core/retention/test_purge.py:214-220` shows `MockPayloadStore.delete` only returns `False` when the ref is missing, so deletion failures for existing refs cannot occur in tests
- `tests/core/retention/test_purge.py:503-521` and `tests/core/retention/test_purge.py:575-594` cover only success and skipped cases; no test asserts `failed_refs` for a ref that exists but fails to delete
- `src/elspeth/core/retention/purge.py:294-301` contains the `failed_refs` branch that should be validated
- Code snippet from `tests/core/retention/test_purge.py:214-220`:
```python
def delete(self, content_hash: str) -> bool:
    self.delete_calls.append(content_hash)
    if content_hash in self._storage:
        del self._storage[content_hash]
        return True
    return False
```

## Impact

- Failure-path behavior is unverified, so regressions could misclassify failed deletions as successes or skips
- Bugs in payload deletion handling could slip through without detection
- Creates false confidence that `failed_refs` reporting works

## Root Cause Hypothesis

- Mock implementation is tuned for happy-path behavior and lacks failure injection
- Test coverage focuses on success and skip scenarios only

## Recommended Fix

- Add a test-specific payload store stub that returns `exists=True` and `delete=False` for a chosen ref; assert `failed_refs == [ref]`, `deleted_count == 0`, `skipped_count == 0`
- Alternatively extend `MockPayloadStore` to allow per-ref failure injection and use it in a new test
- Priority justification: covers an explicit error-handling branch in core retention logic
---
# Test Defect Report

## Summary

- `test_purge_measures_duration` uses a vacuous assertion (`>= 0`) that does not validate the timing logic

## Severity

- Severity: trivial
- Priority: P3

## Category

- Weak Assertions

## Evidence

- `tests/core/retention/test_purge.py:596-609` asserts only `result.duration_seconds >= 0`, which is always true for a non-negative float
- `src/elspeth/core/retention/purge.py:287-306` computes duration from `perf_counter`, but the test does not verify the delta
- Code snippet from `tests/core/retention/test_purge.py:596-609`:
```python
result = manager.purge_payloads([ref])
assert result.duration_seconds >= 0
```

## Impact

- Test will pass even if duration is hardcoded or incorrectly computed
- Timing regressions in `purge_payloads` would go unnoticed
- Provides minimal confidence in the timing behavior

## Root Cause Hypothesis

- Avoided time-dependent assertions without replacing them with a deterministic check
- Pattern of asserting only field existence/value range instead of correctness

## Recommended Fix

- Monkeypatch `elspeth.core.retention.purge.perf_counter` to return fixed values (e.g., `10.0` then `12.5`) and assert `duration_seconds == 2.5`
- Use `pytest.monkeypatch` or `unittest.mock.patch` in `test_purge_measures_duration`
- Priority justification: low risk but strengthens a currently ineffective assertion
