# Test Defect Report

## Summary

- Missing Tier 1 corruption coverage for `get_row_data`; corrupted payload bytes (IntegrityError/invalid JSON) behavior is untested

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Tier 1 Corruption Tests

## Evidence

- `tests/core/landscape/test_recorder_row_data.py:113` only stores valid JSON payloads, so corruption paths are never exercised in this file.
```python
test_data = {"field": "value"}
payload_ref = payload_store.store(json.dumps(test_data).encode())
```
- `src/elspeth/core/landscape/recorder.py:1901` decodes and parses payload bytes, but only catches `KeyError`; JSON decode or integrity errors will propagate and are not asserted by any test in this file.
```python
payload_bytes = self._payload_store.retrieve(row.source_data_ref)
data = json.loads(payload_bytes.decode("utf-8"))
...
except KeyError:
    return RowDataResult(state=RowDataState.PURGED, data=None)
```
- `src/elspeth/core/payload_store.py:136` raises `IntegrityError` on hash mismatch (Tier 1 corruption), but `tests/core/landscape/test_recorder_row_data.py` has no test to ensure this failure is surfaced.
```python
if not hmac.compare_digest(actual_hash, content_hash):
    raise IntegrityError(...)
```

## Impact

- Corrupted payloads (hash mismatch or invalid JSON) could be silently mishandled or masked by future changes without test detection.
- Violates the Tier 1 “crash on anomaly” expectation for audit data if regressions accidentally swallow these errors.
- Creates false confidence that `get_row_data` handles all audit integrity scenarios.

## Root Cause Hypothesis

- Tests were written to cover the explicit state enum transitions only, without extending to corruption/error paths mandated by the auditability standard.

## Recommended Fix

- Add explicit corruption-path tests in `tests/core/landscape/test_recorder_row_data.py`:
  - Tamper with the stored payload bytes (after `store`) so `FilesystemPayloadStore.retrieve` raises `IntegrityError`, and assert the exception propagates.
  - Store non-JSON bytes (with correct hash) and assert `json.JSONDecodeError` propagates from `get_row_data`.
- Use `pytest.raises` to enforce Tier 1 crash-on-corruption behavior; this is a P1 change because it protects audit integrity regressions on a critical path.
