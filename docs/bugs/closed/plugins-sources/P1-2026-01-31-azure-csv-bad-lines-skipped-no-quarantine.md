# Bug Report: CSV bad lines skipped without quarantine in AzureBlobSource

## Summary

- `AzureBlobSource` uses `on_bad_lines="warn"` which causes pandas to silently skip malformed CSV lines with only a warning. No quarantine record is created, violating "every row reaches terminal state".

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/azure/blob_source.py:398` - `on_bad_lines="warn"`
- Pandas skips malformed lines and logs warning, but no audit trail entry
- Violates CLAUDE.md:637-647 "every row reaches exactly one terminal state"

## Impact

- User-facing impact: Rows silently disappear from processing
- Data integrity / security impact: No audit record of skipped rows
- Performance or cost impact: None

## Root Cause Hypothesis

- Using pandas default behavior for bad lines instead of capturing them for quarantine.

## Proposed Fix

- Code changes:
  - Use custom `on_bad_lines` handler to capture malformed lines
  - Create quarantine records for each bad line with the raw content and error
  - Or: Use `on_bad_lines="error"` and handle at source level
- Tests to add/update:
  - Add test with malformed CSV row, verify quarantine record created

## Acceptance Criteria

- Malformed CSV lines are captured and quarantined, not silently skipped
- Each bad line has an audit trail entry explaining the issue
