# Bug Report: Azure batch does not record full request/response payloads

## Summary

- AzureBatchLLMTransform records only metadata for the batch JSONL upload and output download, not the actual JSONL content, violating the audit requirement to capture full external requests and responses.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any batch run using azure_batch_llm

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Run `azure_batch_llm` with any input rows.
2. Inspect the call records for `files.create` and `files.content` in Landscape.

## Expected Behavior

- Full JSONL payload (prompt requests) and output content are recorded, either inline or via payload store references, per auditability standard.

## Actual Behavior

- Only metadata like content_size/content_length is recorded; the JSONL body itself is not persisted in the audit trail.

## Evidence

- Upload request records only metadata in `src/elspeth/plugins/llm/azure_batch.py:392` and `src/elspeth/plugins/llm/azure_batch.py:405`.
- Download response records only content length in `src/elspeth/plugins/llm/azure_batch.py:584` and `src/elspeth/plugins/llm/azure_batch.py:592`.

## Impact

- User-facing impact: explain/replay cannot reconstruct prompts or batch outputs.
- Data integrity / security impact: audit trail is incomplete; violates "full request/response recorded" policy.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Request/response payloads were intentionally omitted to reduce storage but no payload store reference was used.

## Proposed Fix

- Code changes (modules/files):
  - Store JSONL input and output in payload store (or attach to request_data/response_data) and record `request_ref`/`response_ref`.
- Config or schema changes: consider a size threshold to route large payloads to payload store.
- Tests to add/update:
  - Validate that calls for batch upload/download include payload references or content hashes.
- Risks or migration steps:
  - Audit tables may grow; ensure payload retention policies apply.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md "External calls - Full request AND response recorded".
- Observed divergence: only metadata recorded for batch I/O.
- Reason (if known): likely storage concerns; not implemented.
- Alignment plan or decision needed: decide payload storage strategy for large JSONL.

## Acceptance Criteria

- Batch upload/download calls record full request/response content (or payload references) and hashes in Landscape.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_batch.py -v`
- New tests required: yes, audit payload recording test for JSONL upload/download.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard
