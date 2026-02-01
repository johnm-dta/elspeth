# Bug Report: JSONFormatter silently coerces unsupported types and allows NaN/Infinity

## Summary

- `JSONFormatter` uses `json.dumps(record, default=str)` without `allow_nan=False`, allowing NaN/Infinity values and silently coercing unsupported types to strings in audit exports.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/core/landscape/formatters.py:107` - `json.dumps(record, default=str)` with no `allow_nan=False`
- `default=str` coerces any unknown type to string instead of failing
- CLAUDE.md requires canonical JSON (RFC 8785) which rejects NaN/Infinity

## Impact

- User-facing impact: Audit exports may contain invalid JSON (NaN/Infinity)
- Data integrity / security impact: Violates canonical JSON policy; inconsistent parsing across tools
- Performance or cost impact: None

## Root Cause Hypothesis

- Should use `canonical_json()` or at minimum `json.dumps(..., allow_nan=False)` with explicit serializers.

## Proposed Fix

- Code changes:
  - Replace `json.dumps(record, default=str)` with `canonical_json(record)` in JSONFormatter
  - Or use `json.dumps(..., allow_nan=False)` with explicit type handlers
- Tests to add/update:
  - Add test with NaN value, assert export fails or value is properly handled

## Acceptance Criteria

- JSONFormatter rejects NaN/Infinity values
- Unknown types cause explicit errors rather than silent string coercion
