# Bug Report: Azure plugin docstring references non-existent plugin names

## Summary

- Module docstring shows `azure_blob_source` and `azure_blob_sink` but actual plugin name is `azure_blob` for both.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/azure/__init__.py:12-13` - shows `azure_blob_source`, `azure_blob_sink`
- `src/elspeth/plugins/azure/blob_source.py:252` - actual name is `azure_blob`
- `src/elspeth/plugins/azure/blob_sink.py:234` - actual name is `azure_blob`

## Proposed Fix

- Update docstring to show correct plugin name

## Acceptance Criteria

- Documentation matches actual plugin names

## Verification (2026-02-01)

**Status: FIXED**

- Updated `src/elspeth/plugins/azure/__init__.py` docstring to use `azure_blob` for both source and sink
