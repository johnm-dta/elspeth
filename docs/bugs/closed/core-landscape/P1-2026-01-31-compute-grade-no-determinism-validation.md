# Bug Report: compute_grade silently accepts invalid node determinism values

## Summary

- `compute_grade()` only checks for specific values in the `non_reproducible` set without validating that all determinism values are valid enum members. Invalid values like `"garbage"` would be treated as reproducible.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/core/landscape/reproducibility.py:56-77` - `compute_grade()` only checks for specific values in `non_reproducible` set
- Invalid enum values are silently treated as reproducible
- Tier 1 audit data should crash on invalid enum values

## Impact

- User-facing impact: Reproducibility grade could be incorrect
- Data integrity / security impact: Invalid determinism values in audit trail go undetected
- Performance or cost impact: None

## Root Cause Hypothesis

- Missing validation that all determinism values are valid `Determinism` enum members before computing grade.

## Proposed Fix

- Code changes:
  - Fetch all distinct determinism values and validate each is a valid `Determinism` enum member
  - Raise ValueError for invalid values before computing grade
- Tests to add/update:
  - Add test with invalid determinism value, assert ValueError raised

## Acceptance Criteria

- Invalid determinism values cause immediate failure
- Valid determinism values continue to compute correct grades
