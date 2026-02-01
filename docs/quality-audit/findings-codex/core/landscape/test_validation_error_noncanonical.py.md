# Test Defect Report

## Summary

- Tests for non-canonical validation errors only assert `error_id` and never verify `validation_errors` records (row_hash, row_data_json, error, schema_mode, destination), leaving audit trail integrity untested for most cases.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/core/landscape/test_validation_error_noncanonical.py:50` only checks returned token fields for primitive int and never inspects audit records:
```python
token = ctx.record_validation_error(...)
assert token.error_id is not None
```
- `tests/core/landscape/test_validation_error_noncanonical.py:70` and `tests/core/landscape/test_validation_error_noncanonical.py:88` repeat the same minimal assertion for string/list inputs.
- `tests/core/landscape/test_validation_error_noncanonical.py:106` and `tests/core/landscape/test_validation_error_noncanonical.py:125` only assert `error_id` for NaN/Infinity cases (no row_hash/row_data_json checks).
- `tests/core/landscape/test_validation_error_noncanonical.py:200` checks only uniqueness of `error_id`s for multiple rows, not actual recorded data.
- `src/elspeth/core/landscape/recorder.py:2386` shows the recorder writes `row_hash`, `row_data_json`, `error`, `schema_mode`, and `destination`, but the tests above never validate those fields.

## Impact

- A regression that writes incorrect `row_hash`, fails to store `row_data_json`, or misrecords `schema_mode`/`destination` would pass these tests, undermining audit trail integrity.
- Non-canonical data handling is a Tier-3 boundary; missing verification creates false confidence in the audit record.

## Root Cause Hypothesis

- Tests focus on API return values rather than validating the persisted audit trail, despite the presence of recorder query helpers.

## Recommended Fix

- In each test, query the audit trail via `LandscapeRecorder.get_validation_errors_for_run` or `LandscapeRecorder.get_validation_errors_for_row` and assert:
  - `row_hash` matches `stable_hash` for canonicalizable data and `repr_hash` for NaN/Infinity.
  - `row_data_json` is canonical JSON for primitives and `NonCanonicalMetadata` JSON for non-finite values.
  - `error`, `schema_mode`, `destination`, `node_id` match inputs.
- Example assertion pattern (in this test file):
```python
records = recorder.get_validation_errors_for_run("test-run")
assert records[-1].error == "Row contains NaN"
assert "__canonical_error__" in json.loads(records[-1].row_data_json)
```
- Priority justified because audit trail correctness is core to ELSPETH’s safety contract.
---
# Test Defect Report

## Summary

- Test accesses private `recorder._db` directly instead of using public query helpers, coupling tests to internal implementation.

## Severity

- Severity: minor
- Priority: P3

## Category

- Infrastructure Gaps

## Evidence

- `tests/core/landscape/test_validation_error_noncanonical.py:183` uses a private attribute:
```python
with recorder._db.connection() as conn:
    result = conn.execute(...)
```
- `src/elspeth/core/landscape/recorder.py:2499` provides public helpers `get_validation_errors_for_row` and `get_validation_errors_for_run` that could be used instead.

## Impact

- Refactors to the recorder’s internal DB layer can break tests unnecessarily, reducing maintainability and masking real regressions.

## Root Cause Hypothesis

- Convenience or unawareness of existing public query methods in `LandscapeRecorder`.

## Recommended Fix

- Replace direct `_db` access with `recorder.get_validation_errors_for_row` or `recorder.get_validation_errors_for_run` in `tests/core/landscape/test_validation_error_noncanonical.py:161`.
- Example:
```python
records = recorder.get_validation_errors_for_run("test-run")
row_data = json.loads(records[-1].row_data_json)
```
- Priority P3 since it is a maintainability issue, not a functional correctness gap.
