# Test Defect Report

## Summary

- Weak assertions: the test only validates type fidelity on the first unprocessed row, leaving remaining returned rows unchecked.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/checkpoint/test_recovery_type_fidelity.py:169` captures `unprocessed` and asserts length only.
- `tests/core/checkpoint/test_recovery_type_fidelity.py:173` inspects only `unprocessed[0]` and never checks `unprocessed[1]`.
- `tests/core/checkpoint/test_recovery_type_fidelity.py:187` asserts type restoration for only the first row.
- Code snippet showing only the first row is validated:
```python
unprocessed = recovery_manager.get_unprocessed_row_data(run_id, payload_store, source_schema_class=source_schema_class)

assert len(unprocessed) == 2  # Rows 1 and 2 are unprocessed

row_id, row_index, row_data_restored = unprocessed[0]
...
assert isinstance(timestamp_restored, datetime)
assert isinstance(amount_restored, Decimal)
```

## Impact

- A regression that restores types correctly for the first row but fails on later rows would still pass this test.
- Incomplete coverage can mask row-specific decoding/validation bugs, giving false confidence in resume type fidelity.

## Root Cause Hypothesis

- Test was written as a focused reproduction of the original bug fix and only validated a single representative row.

## Recommended Fix

- Iterate over all `unprocessed` entries and assert expected types/values for each row, not just the first. Example:
```python
expected = {
    1: (2, datetime(2024, 1, 2, 12, 0, tzinfo=UTC), Decimal("200.75")),
    2: (3, datetime(2024, 1, 3, 12, 0, tzinfo=UTC), Decimal("300.25")),
}
for _row_id, row_index, row_data in unprocessed:
    exp_id, exp_ts, exp_amt = expected[row_index]
    assert row_data["id"] == exp_id
    assert isinstance(row_data["timestamp"], datetime)
    assert isinstance(row_data["amount"], Decimal)
    assert row_data["timestamp"] == exp_ts
    assert row_data["amount"] == exp_amt
```
- Priority justification: this is the only direct test of datetime/Decimal restoration; covering all returned rows prevents row-specific regressions from slipping through.
