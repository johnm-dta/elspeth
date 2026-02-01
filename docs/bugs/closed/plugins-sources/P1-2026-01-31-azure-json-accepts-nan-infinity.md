# Bug Report: AzureBlobSource accepts NaN/Infinity in JSON input

## Summary

- `AzureBlobSource` uses `json.loads()` without `parse_constant` to reject NaN/Infinity. Unlike `JSONSource`, it allows non-canonical values into the pipeline which can break downstream canonical hashing.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/azure/blob_source.py:450, 502` - `json.loads(...)` without `parse_constant`
- `JSONSource` uses `parse_constant` to reject NaN/Infinity at source boundary
- `AzureBlobSource` does not, allowing non-canonical values into pipeline
- These can later break `canonical_json()` which rejects NaN/Infinity

## Impact

- User-facing impact: Pipeline crashes at transform/sink instead of at source
- Data integrity / security impact: Non-canonical values could reach audit trail before rejection
- Performance or cost impact: Wasted processing of rows that will fail downstream

## Root Cause Hypothesis

- `AzureBlobSource` JSON parsing was not aligned with `JSONSource` validation patterns.

## Proposed Fix

- Code changes:
  - Add `parse_constant` handler to `json.loads()` calls that raises on NaN/Infinity
  - Or: Validate parsed data immediately after loading
- Tests to add/update:
  - Add test with JSON containing NaN, verify rejected at source

## Acceptance Criteria

- NaN/Infinity values in JSON input are rejected at source boundary
- Consistent behavior between `JSONSource` and `AzureBlobSource`
