# Bug Report: AzureBlobSource JSON array errors crash instead of quarantining

## Summary

- `AzureBlobSource` raises `ValueError` on JSON structure errors (not an array, not objects, etc.) instead of quarantining the malformed input. This crashes the run instead of handling external data gracefully.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/azure/blob_source.py:452-463`:
  - Lines 452, 454, 458, 462 all `raise ValueError` on JSON errors
  - This crashes the run instead of quarantining
- Violates Tier-3 trust model: external data should be quarantined, not crash the system

## Impact

- User-facing impact: Single malformed blob crashes entire pipeline
- Data integrity / security impact: None (fails safe, but unnecessarily)
- Performance or cost impact: Wasted run time, requires manual intervention

## Root Cause Hypothesis

- Error handling treats external JSON errors as fatal instead of quarantinable.

## Proposed Fix

- Code changes:
  - Catch JSON structure errors and yield quarantine entries instead of raising
  - Record the raw content and error reason in quarantine
  - Allow pipeline to continue with other valid data
- Tests to add/update:
  - Add test with malformed JSON blob, verify quarantine not crash

## Acceptance Criteria

- JSON structure errors result in quarantine, not crash
- Pipeline continues processing valid data after quarantining bad input
