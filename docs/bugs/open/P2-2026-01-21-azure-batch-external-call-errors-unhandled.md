# Bug Report: Azure batch external API failures crash the transform

## Summary

- AzureBatchLLMTransform calls Azure OpenAI batch endpoints without try/except; any network/auth/HTTP error raises and crashes the pipeline instead of returning TransformResult.error with audit details.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any batch run using azure_batch_llm with invalid creds or network outage

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure `azure_batch_llm` with an invalid API key or block network access.
2. Run a batch submission or resume.
3. Observe unhandled exceptions during file upload, batch create, status check, or output download.

## Expected Behavior

- External API errors are caught, recorded, and returned as TransformResult.error with retryable classification where appropriate.

## Actual Behavior

- Exceptions propagate out of the transform, crashing the run without structured error routing.

## Evidence

- No error handling around `client.files.create` in `src/elspeth/plugins/llm/azure_batch.py:401`.
- No error handling around `client.batches.create` in `src/elspeth/plugins/llm/azure_batch.py:421`.
- No error handling around `client.batches.retrieve` in `src/elspeth/plugins/llm/azure_batch.py:483`.
- No error handling around `client.files.content` in `src/elspeth/plugins/llm/azure_batch.py:591`.

## Impact

- User-facing impact: pipeline crashes instead of routing failures to on_error sink.
- Data integrity / security impact: missing structured error records for external calls.
- Performance or cost impact: retries require full reruns.

## Root Cause Hypothesis

- External API calls were implemented without the standard try/except wrappers used in other LLM transforms.

## Proposed Fix

- Code changes (modules/files): wrap Azure client calls with try/except, return TransformResult.error, and record call errors via ctx.record_call.
- Config or schema changes: N/A
- Tests to add/update:
  - Simulate Azure API failures for submit/retrieve/content and assert TransformResult.error.
- Risks or migration steps:
  - Ensure retryable classification aligns with RetryManager usage.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md external system boundaries should be wrapped.
- Observed divergence: external calls can crash the transform.
- Reason (if known): not implemented for batch path.
- Alignment plan or decision needed: apply same error-handling pattern as other LLM transforms.

## Acceptance Criteria

- All Azure batch API failures produce structured TransformResult.error (not exceptions), with audit call records.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_batch.py -v`
- New tests required: yes, failure-path tests for upload/create/retrieve/content.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard
