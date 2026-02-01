# Test Defect Report

## Summary

- using-quality-engineering audit: no test covers the empty-schema guard in `get_unprocessed_row_data`, leaving the silent data-loss path unverified.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Edge Cases

## Evidence

- `src/elspeth/core/checkpoint/recovery.py:206` defines the guard that should raise when schema validation drops all fields.
```python
if degraded_data and not row_data:
    raise ValueError(
        f"Resume failed for row {row_id}: Schema validation returned empty data "
```
- `tests/core/checkpoint/test_recovery_row_data.py:31` shows the tests always use the dynamic schema, which keeps extras and will not trigger the empty-schema guard.
```python
from elspeth.plugins.schema_factory import _create_dynamic_schema

mock_schema = _create_dynamic_schema("MockSchema")
```
- `src/elspeth/plugins/schema_factory.py:83` shows the dynamic schema uses `extra="allow"`, so this file never exercises the empty-schema error path.
```python
return create_model(
    name,
    __base__=PluginSchema,
    __config__=ConfigDict(extra="allow"),
)
```

## Impact

- A regression that removes or weakens the empty-schema guard would still pass this test file.
- Resume could silently discard all row fields if a schema is empty or misdeclared, breaking auditability and lineage.
- The test suite gives false confidence that resume guards against data loss.

## Root Cause Hypothesis

- Tests focus on the main retrieval path and missing payload handling, but not on the guard added for Null/empty schema cases.
- Reuse of `_create_dynamic_schema` for convenience avoids the failure mode the guard is meant to catch.

## Recommended Fix

- Add a test in `tests/core/checkpoint/test_recovery_row_data.py` that uses a schema with no fields (e.g., a `PluginSchema` subclass with no attributes) and asserts `ValueError` with "Schema validation returned empty data".
- Reuse `run_with_checkpoint_and_payloads` to supply non-empty payloads so `degraded_data` is truthy and the guard is exercised.
- Priority: protects a core auditability invariant (no silent data loss on resume).
---
# Test Defect Report

## Summary

- Row ID correctness is only type-checked; expected row IDs for unprocessed rows are never asserted.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/checkpoint/test_recovery_row_data.py:52` only verifies row IDs are strings, not the expected values.
```python
row_ids = [item[0] for item in row_data_list]
assert all(isinstance(r, str) for r in row_ids)
```
- `tests/core/checkpoint/conftest.py:111` defines deterministic row IDs (`row-003`, `row-004` for unprocessed rows) that should be asserted.
```python
row_id = f"row-{i:03d}"
```

## Impact

- A bug returning incorrect row IDs (e.g., stringified indices or constant IDs) would pass this test.
- Lineage integrity could be broken without detection.
- Test confidence is inflated because critical identifiers are not validated.

## Root Cause Hypothesis

- Emphasis on row_index and row_data checks led to assuming row_id correctness from the DB.
- Assertion was kept minimal to avoid coupling to fixture details.

## Recommended Fix

- Strengthen the assertion to compare against expected IDs, e.g., `assert row_ids == ["row-003", "row-004"]`.
- Alternatively, compute expected IDs from the fixture pattern to keep the test intention explicit.
- Priority: improves correctness checks for lineage-critical identifiers.
