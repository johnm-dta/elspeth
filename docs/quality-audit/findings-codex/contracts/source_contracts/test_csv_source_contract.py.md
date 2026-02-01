# Test Defect Report

## Summary

- Contract tests validate SourceRow outputs but never verify audit trail recording for validation failures or parse errors in CSVSource.

## Severity

- Severity: major
- Priority: P1

## Category

- [Missing Audit Trail Verification]

## Evidence

- `tests/contracts/source_contracts/test_csv_source_contract.py:146-164` constructs a bare `PluginContext` and only checks row counts/quarantine flags; no audit/Landscape assertions are made.
- `tests/contracts/source_contracts/test_csv_source_contract.py:180-201` discard-mode test only checks output rows and does not verify any audit trail recording.
- `src/elspeth/plugins/sources/csv_source.py:135-199` explicitly calls `ctx.record_validation_error(...)` for parse/validation errors, which is a critical audit side effect that is untested here.

```python
# tests/contracts/source_contracts/test_csv_source_contract.py:146-164
ctx = PluginContext(run_id="test", config={})
rows = list(source.load(ctx))
valid_rows = [r for r in rows if not r.is_quarantined]
quarantined_rows = [r for r in rows if r.is_quarantined]
assert len(quarantined_rows) == 1
```

```python
# src/elspeth/plugins/sources/csv_source.py:135-199
ctx.record_validation_error(
    row=row,
    error=str(e),
    schema_mode=self._schema_config.mode or "dynamic",
    destination=self._on_validation_failure,
)
```

## Impact

- A regression that drops or misrecords validation errors in the Landscape audit trail would pass these contract tests.
- This creates false confidence in audit completeness for a core compliance requirement.
- Missing verification means schema_mode/destination/hash integrity could be wrong without detection.

## Root Cause Hypothesis

- Contract tests focus on SourceRow behavior but omit audit side effects and avoid Landscape setup for simplicity.
- PluginContext is instantiated without a LandscapeRecorder, so audit writes cannot be asserted.

## Recommended Fix

- In this file, add audit-backed contract checks for invalid rows using a real in-memory Landscape DB and recorder, not a mock.
- Create a run and (optionally) a node, pass `landscape=recorder` and `node_id` into `PluginContext`, then assert validation_errors rows match expected schema_mode, destination, and row_hash/canonical JSON.
- Example pattern:

```python
from elspeth.core.canonical import CANONICAL_VERSION, stable_hash
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

db = LandscapeDB.in_memory()
recorder = LandscapeRecorder(db)
run = recorder.begin_run(config={}, canonical_version=CANONICAL_VERSION, run_id="test")

ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder, node_id="csv_source")
rows = list(source.load(ctx))

row_hash = stable_hash({"id": "not_an_int", "name": "Bob"})
errors = recorder.get_validation_errors_for_row(run.run_id, row_hash)
assert len(errors) == 1
assert errors[0].schema_mode == "strict"
assert errors[0].destination == "quarantine_sink"
```
---
# Test Defect Report

## Summary

- Quarantined-row assertions are too weak: they only assert `row is not None`, despite the test comment stating that original row data is preserved.

## Severity

- Severity: minor
- Priority: P2

## Category

- [Weak Assertions]

## Evidence

- `tests/contracts/source_contracts/test_csv_source_contract.py:160-164` only checks that the quarantined row exists, not that it preserves the original invalid data.

```python
# tests/contracts/source_contracts/test_csv_source_contract.py:160-164
q_row = quarantined_rows[0]
assert q_row.quarantine_error is not None
assert q_row.quarantine_destination == "quarantine_sink"
assert q_row.row is not None  # Original row data preserved
```

## Impact

- If CSVSource accidentally drops or mutates invalid row data, this test still passes.
- This weakens audit/debugging confidence for quarantined rows and could hide regressions in error handling.

## Root Cause Hypothesis

- The test comment indicates intended verification, but the assertion was implemented as a presence check instead of a content check.
- Likely a “good enough” assertion choice during initial test scaffolding.

## Recommended Fix

- Strengthen assertions to verify the actual quarantined row content and error details.
- Example:

```python
assert q_row.is_quarantined is True
assert q_row.row == {"id": "not_an_int", "name": "Bob"}
assert "id" in q_row.quarantine_error.lower()
```

This keeps the contract precise and protects against row-data loss regressions.
