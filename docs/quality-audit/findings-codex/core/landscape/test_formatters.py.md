# Test Defect Report

## Summary

- JSONFormatter tests use weak/partial assertions (datetime and nested list content), allowing loss of time/timezone or event fields to pass unnoticed.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/landscape/test_formatters.py:142` and `tests/core/landscape/test_formatters.py:153` only check a date substring, not full timestamp or timezone:
```python
def test_json_formatter_handles_datetime_via_default(self) -> None:
    ...
    parsed = json.loads(output)
    assert "2024-01-15" in parsed["timestamp"]
```
- `tests/core/landscape/test_formatters.py:168` and `tests/core/landscape/test_formatters.py:183` validate only list length and the first element, leaving other fields/entries unchecked:
```python
parsed = json.loads(output)
assert len(parsed["events"]) == 2
assert parsed["events"][0]["type"] == "click"
```

## Impact

- Regressions that drop timestamp time or timezone detail can pass unnoticed.
- Partial data loss or mutation in nested event lists can slip through.
- Creates false confidence in JSON export fidelity for audit records.

## Root Cause Hypothesis

- Tests prioritize shape/presence checks over exact value preservation for critical fields.

## Recommended Fix

- Assert exact datetime serialization and full nested list content. Update `tests/core/landscape/test_formatters.py:142` and `tests/core/landscape/test_formatters.py:168` to compare full values, e.g.:
```python
expected_ts = str(datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC))
assert parsed["timestamp"] == expected_ts
assert parsed["events"] == [
    {"type": "click", "target": "button"},
    {"type": "scroll", "position": 100},
]
```
- Priority P2 because export fidelity is part of auditability guarantees.
