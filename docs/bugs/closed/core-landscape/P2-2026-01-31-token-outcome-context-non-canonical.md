# Bug Report: Token Outcome Context Stored with json.dumps (Non-Canonical JSON)

## Summary

- `record_token_outcome` serializes `context` with `json.dumps`, allowing NaN/Infinity and bypassing canonical JSON enforcement.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/landscape/recorder.py:2433` - `context_json = json.dumps(context)`
- Uses `json.dumps` instead of `canonical_json`, allowing NaN/Infinity
- Violates CLAUDE.md canonical JSON policy

## Impact

- User-facing impact: Audit exports may contain invalid JSON
- Data integrity: Inconsistent serialization across audit trail

## Proposed Fix

- Replace `json.dumps(context)` with `canonical_json(context)`

## Acceptance Criteria

- `record_token_outcome` uses canonical JSON and rejects NaN/Infinity

## Verification (2026-02-01)

**Status: STILL VALID**

- `record_token_outcome()` still serializes `context` via plain `json.dumps()`. (`src/elspeth/core/landscape/recorder.py:2433`)

## Resolution (2026-02-02)

**Status: FIXED**

### Fix Applied

Changed `recorder.py:2433` from:
```python
context_json = json.dumps(context) if context is not None else None
```
to:
```python
context_json = canonical_json(context) if context is not None else None
```

### Tests Added

Added `TestTokenOutcomeCanonicalJson` class in `tests/core/landscape/test_token_outcome_constraints.py`:
- `test_context_with_nan_raises_value_error` - Verifies NaN in context raises ValueError
- `test_context_with_infinity_raises_value_error` - Verifies Infinity in context raises ValueError
- `test_context_with_valid_data_succeeds` - Verifies normal data works fine

### Verification

- All 3 new tests pass
- All 493 existing landscape tests pass
- No regressions introduced
