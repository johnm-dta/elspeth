# Bug Report: TokenOutcomeRepository coerces invalid is_terminal values

## Summary

- `TokenOutcomeRepository` uses `is_terminal=row.is_terminal == 1` which treats any non-1 value (including NULL, 2, 99) as False without validation. This violates Tier 1 crash-on-anomaly rules.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/core/landscape/repositories.py:478` - `is_terminal=row.is_terminal == 1`
- Any non-1 value (NULL, 2, 99) is coerced to False without error
- Should validate `row.is_terminal in (0, 1)` and raise otherwise

## Impact

- User-facing impact: Corrupted is_terminal values silently coerced
- Data integrity / security impact: Audit data corruption goes undetected
- Performance or cost impact: None

## Root Cause Hypothesis

- Missing validation that `is_terminal` is exactly 0 or 1 before coercing to boolean.

## Proposed Fix

- Code changes:
  - Add validation: `if row.is_terminal not in (0, 1): raise ValueError(...)`
  - Then coerce: `is_terminal = row.is_terminal == 1`
- Tests to add/update:
  - Add test with invalid is_terminal value (e.g., 2), assert ValueError

## Acceptance Criteria

- Invalid is_terminal values (not 0 or 1) cause immediate failure
- Valid values (0, 1) continue to work correctly
