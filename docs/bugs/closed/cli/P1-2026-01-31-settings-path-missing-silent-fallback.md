# Bug Report: Explicit --settings path silently ignored when missing

## Summary

- `resolve_database_url` ignores a user-supplied `settings_path` if the file does not exist and silently falls back to `./settings.yaml`, which can point to a different database than the user intended.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/cli_helpers.py:108-114` - explicit `settings_path` is only used when `settings_path.exists()` is true.
- `src/elspeth/cli_helpers.py:116-124` - when the explicit path is missing, the function falls through and loads `settings.yaml` if it exists, with no error mentioning the missing explicit path.
- There is no branch that raises for a missing *explicit* `settings_path`, so a typo silently changes configuration precedence.

## Impact

- User-facing impact: User can unknowingly query or operate on the wrong audit database when they mistype `--settings`
- Data integrity / security impact: Audit lineage may be reported from the wrong run database, undermining traceability
- Performance or cost impact: Wasted investigation time

## Root Cause Hypothesis

- The function treats a non-existent explicit `settings_path` the same as "not provided," violating configuration precedence.

## Proposed Fix

- Code changes:
  - Add explicit check: if `settings_path` is provided and does not exist, raise `ValueError` immediately
- Tests to add/update:
  - Add unit test for `resolve_database_url` that passes non-existent `settings_path` and asserts `ValueError`

## Acceptance Criteria

- Providing `--settings` with a non-existent file results in a clear `ValueError` and no fallback to default settings
- Existing behavior when `--settings` is valid or omitted remains unchanged

## Verification (2026-02-01)

**Status: ALREADY FIXED**

- Bug was filed 2026-01-31, but fix was merged 2026-01-29 in commit `8ab8fb36` ("feat(cli): add database resolution helpers")
- Commit message explicitly states: "Fail fast if database file doesn't exist"
- Current code at `src/elspeth/cli_helpers.py:109-111` raises `ValueError("Settings file not found: {settings_path}")` when explicit path doesn't exist
- Test `test_raises_when_explicit_settings_path_missing` in `tests/cli/test_cli_helpers_db.py` verifies this behavior
- All 13 tests in test file pass, confirming fix is in place
- Previous verification was incorrect - likely examined stale line numbers

## Closure

- **Closed by:** Claude (systematic debugging investigation)
- **Closure date:** 2026-02-01
- **Resolution:** Already fixed prior to bug filing
