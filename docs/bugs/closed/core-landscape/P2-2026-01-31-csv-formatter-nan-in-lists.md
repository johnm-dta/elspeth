# Bug Report: CSVFormatter allows NaN/Infinity in list serialization

## Summary

- `CSVFormatter` uses `json.dumps(value)` for list fields without `allow_nan=False`, allowing NaN/Infinity in CSV exports.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/landscape/formatters.py:215-217` - `json.dumps(value)` without `allow_nan=False`.

## Impact

- User-facing impact: CSV exports may contain non-standard JSON
- Data integrity: Inconsistent serialization

## Proposed Fix

- Use `json.dumps(..., allow_nan=False)` or `canonical_json()`

## Acceptance Criteria

- List values with NaN/Infinity are rejected or handled consistently

## Verification (2026-02-01)

**Status: STILL VALID**

- `CSVFormatter` still serializes list values via `json.dumps()` without NaN/Infinity rejection. (`src/elspeth/core/landscape/formatters.py:215-217`)

## Resolution (2026-02-02)

**Status: FIXED**

### Root Cause

`CSVFormatter.flatten()` used raw `json.dumps(value)` on list fields, bypassing the existing `serialize_datetime()` helper which already rejects NaN/Infinity.

### Fix Applied

Changed `formatters.py:220` from:
```python
result[full_key] = json.dumps(value)
```
to:
```python
result[full_key] = json.dumps(serialize_datetime(value))
```

This uses the existing `serialize_datetime()` function which:
- Rejects NaN with `ValueError: "NaN values are not allowed in audit data"`
- Rejects Infinity with `ValueError: "Infinity values are not allowed in audit data"`
- Converts datetime objects to ISO strings

### Tests Added

Added to `TestCSVFormatter` in `tests/core/landscape/test_formatters.py`:
- `test_csv_formatter_rejects_nan_in_list` - Verifies NaN in lists raises ValueError
- `test_csv_formatter_rejects_infinity_in_list` - Verifies Infinity in lists raises ValueError
- `test_csv_formatter_rejects_nested_nan_in_list` - Verifies NaN in nested list structures raises
- `test_csv_formatter_handles_valid_list_with_floats` - Verifies normal floats work fine

### Verification

- All 34 formatter tests pass
- No regressions introduced
