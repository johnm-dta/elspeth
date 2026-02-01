# Test Defect Report

## Summary

- Weak assertions on timestamp normalization allow incorrect UTC conversion to pass undetected

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/test_canonical.py:211-218` only checks for a UTC suffix, not the actual converted timestamp, so a bug that keeps local time but appends `+00:00` would pass:
```python
def test_pandas_timestamp_aware_to_utc_iso(self) -> None:
    ts = pd.Timestamp("2026-01-12 10:30:00", tz="US/Eastern")
    result = _normalize_value(ts)
    # Should be converted to UTC
    assert "+00:00" in result or "Z" in result
    assert type(result) is str
```
- `tests/core/test_canonical.py:279-286` only checks that the date substring exists, not the full ISO timestamp, so missing/incorrect time or timezone could pass:
```python
def test_list_with_mixed_types(self) -> None:
    data = [np.int64(1), pd.Timestamp("2026-01-12"), None]
    result = _normalize_for_canonical(data)
    assert result[0] == 1
    assert "2026-01-12" in result[1]
    assert result[2] is None
```

## Impact

- Allows incorrect timezone conversions to go unnoticed, producing wrong canonical JSON and hashes.
- Undermines audit integrity for timestamped data because canonicalization could silently drift by hours.
- Creates false confidence in correctness of UTC normalization, a critical deterministic step.

## Root Cause Hypothesis

- Tests were written to avoid dealing with exact timezone conversions and DST offsets, trading correctness for simplicity.

## Recommended Fix

- Assert exact expected UTC strings for deterministic inputs.
- Add explicit equality checks for both aware and naive timestamps in these tests.

Example update:
```python
# tests/core/test_canonical.py
def test_pandas_timestamp_aware_to_utc_iso(self) -> None:
    ts = pd.Timestamp("2026-01-12 10:30:00", tz="US/Eastern")
    result = _normalize_value(ts)
    assert result == "2026-01-12T15:30:00+00:00"

def test_list_with_mixed_types(self) -> None:
    data = [np.int64(1), pd.Timestamp("2026-01-12"), None]
    result = _normalize_for_canonical(data)
    assert result[1] == "2026-01-12T00:00:00+00:00"
```
Priority justification: This protects deterministic hashing and audit trail integrity for a core canonicalization path.
