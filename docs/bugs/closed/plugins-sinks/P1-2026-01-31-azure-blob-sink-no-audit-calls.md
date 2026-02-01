# Bug Report: AzureBlobSink does not record Azure Blob SDK calls in audit trail

## Summary

- AzureBlobSink makes external calls to Azure Blob Storage (exists(), upload_blob()) but never records them via `ctx.record_call()`, violating the audit requirement for external call recording.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/azure/blob_sink.py:435-445` - `blob_client.exists()` and `blob_client.upload_blob()` are external calls
- No `ctx.record_call()` invocation anywhere in the sink
- CLAUDE.md requires: "External calls - Full request AND response recorded"

## Impact

- User-facing impact: Audit trail incomplete for Azure Blob operations
- Data integrity / security impact: Cannot trace blob upload operations for compliance
- Performance or cost impact: None

## Root Cause Hypothesis

- Audit call recording was not implemented when the sink was created.

## Proposed Fix

- Code changes:
  - Add `ctx.record_call()` for `blob_client.exists()` and `blob_client.upload_blob()` operations
  - Record operation type, blob name, response status, latency
- Tests to add/update:
  - Add test verifying that Azure Blob operations are recorded in audit trail

## Acceptance Criteria

- All Azure Blob SDK calls are recorded via `ctx.record_call()`
- Audit export includes blob operation details
