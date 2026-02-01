Using skill `using-quality-engineering` (test maintenance patterns) because this is a test quality audit.
# Test Defect Report

## Summary

- `test_all_outcomes_accepted` claims all RowOutcome values are accepted but only exercises three outcomes, leaving most enum members untested

## Severity

- Severity: minor
- Priority: P3

## Category

- Missing Edge Cases

## Evidence

- `tests/engine/test_row_outcome.py:22` hard-codes only COMPLETED/ROUTED/FAILED while the docstring says "All RowOutcome values should work."
```python
for outcome in [RowOutcome.COMPLETED, RowOutcome.ROUTED, RowOutcome.FAILED]:
    result = RowResult(
        token=token,
        final_data={},
        outcome=outcome,
    )
    assert result.outcome == outcome
```
- `src/elspeth/contracts/enums.py:161` defines additional outcomes that are not covered by the test.
```python
FORKED = "forked"
QUARANTINED = "quarantined"
CONSUMED_IN_BATCH = "consumed_in_batch"
COALESCED = "coalesced"
EXPANDED = "expanded"
BUFFERED = "buffered"
```

## Impact

- Regressions affecting non-terminal or less common outcomes (BUFFERED/EXPANDED/COALESCED/etc.) would not be caught by this test, creating false confidence that all outcomes are accepted by RowResult

## Root Cause Hypothesis

- The test likely predates newer RowOutcome values or was copied from an early subset and never updated when the enum expanded

## Recommended Fix

- Iterate over the enum instead of a hard-coded subset (e.g., `for outcome in RowOutcome:`) or explicitly include every RowOutcome member in the list
- Keep the docstring aligned with the actual coverage if only a subset is intended
---
# Test Defect Report

## Summary

- Contract-only tests live under the engine test directory, which misclassifies them as engine tests

## Severity

- Severity: trivial
- Priority: P3

## Category

- Misclassified Tests

## Evidence

- `tests/engine/test_row_outcome.py:3` resides under `tests/engine` but imports only contracts types and never exercises engine code.
```python
from elspeth.contracts import RowOutcome, RowResult, TokenInfo
```

## Impact

- Test suite organization is misleading; running engine tests picks up contract tests, obscuring coverage boundaries and making targeted runs less accurate

## Root Cause Hypothesis

- The test was added near engine work and placed under `tests/engine` for convenience rather than aligning with the contracts test suite

## Recommended Fix

- Move the file to `tests/contracts/test_row_outcome.py` or merge these assertions into existing contracts tests such as `tests/contracts/test_results.py`, keeping naming consistent
