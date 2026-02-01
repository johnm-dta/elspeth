# Bug Report: Azure Batch audit calls omit JSONL payloads

## Summary

- Batch upload/download calls record only size metadata; full JSONL request/response payloads are not captured in the audit trail.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4 @ 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit for `src/elspeth/plugins/llm/azure_batch.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a pipeline with `azure_batch_llm` and run until the batch completes.
2. Inspect the batch upload and download call records in the audit trail.
3. Observe that the recorded request/response do not include JSONL payloads.

## Expected Behavior

- External batch calls record full JSONL request and response payloads (or payload store references) in the audit trail.

## Actual Behavior

- Only `content_size` and `content_length` metadata are recorded; the JSONL payloads are missing.

## Evidence

- `src/elspeth/plugins/llm/azure_batch.py:388` builds `jsonl_content` but it is not passed to `record_call`.
- `src/elspeth/plugins/llm/azure_batch.py:396` constructs `upload_request` with only size metadata; `src/elspeth/plugins/llm/azure_batch.py:407` records only that metadata.
- `src/elspeth/plugins/llm/azure_batch.py:586` and `src/elspeth/plugins/llm/azure_batch.py:594` record only output size metadata.
- `src/elspeth/core/landscape/recorder.py:2066` hashes and stores only the provided `request_data`/`response_data`, so omitted payloads never enter the audit trail.

## Impact

- User-facing impact: `explain()`/replay cannot reconstruct prompts or batch outputs.
- Data integrity / security impact: audit trail violates “full request AND response recorded.”
- Performance or cost impact: Unknown.

## Root Cause Hypothesis

- `record_call` is invoked with metadata-only `request_data`/`response_data`, omitting the JSONL payloads.

## Proposed Fix

- Code changes (modules/files):
  - Include the full JSONL input payload (or parsed request list) in `request_data` when recording `files.create` in `src/elspeth/plugins/llm/azure_batch.py`.
  - Include the full output JSONL (or parsed results list) in `response_data` for `files.content` in `src/elspeth/plugins/llm/azure_batch.py`.
  - If size is a concern, store JSONL content in the payload store and include refs plus content hashes in `request_data`/`response_data`.
- Config or schema changes: Unknown
- Tests to add/update:
  - Add tests in `tests/plugins/llm/test_azure_batch.py` that assert payloads (or payload refs) are recorded for upload/download calls.
- Risks or migration steps:
  - Larger audit payloads; ensure payload store is enabled/retention policies are acceptable.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:23`
- Observed divergence: external calls do not record full request/response payloads.
- Reason (if known): request/response data only contains size metadata.
- Alignment plan or decision needed: record full payloads directly or via payload store refs.

## Acceptance Criteria

- Batch upload/download calls record full request/response payloads (or payload refs) in the audit trail.
- `calls.request_hash`/`calls.response_hash` correspond to the actual payload content.
- `explain()` can reconstruct prompts and batch outputs from the audit trail.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_batch.py`
- New tests required: yes, verify payload recording for batch upload/download.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md:23`
---
# Bug Report: Azure Batch SDK calls lack error handling and error call recording

## Summary

- External Azure SDK calls are not wrapped; exceptions bypass audit recording and crash the pipeline without CallStatus.ERROR entries.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4 @ 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit for `src/elspeth/plugins/llm/azure_batch.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a pipeline with `azure_batch_llm`.
2. Induce an SDK/network failure (e.g., mock `client.files.create` to raise).
3. Run the pipeline and observe the exception and missing call error record.

## Expected Behavior

- External call failures are caught, recorded with CallStatus.ERROR, and surfaced as TransformResult.error (or retried via BatchPendingError).

## Actual Behavior

- Exceptions propagate without error call records; the batch fails with no audit entry for the failed external call.

## Evidence

- `src/elspeth/plugins/llm/azure_batch.py:403` calls `client.files.create` without try/except; `src/elspeth/plugins/llm/azure_batch.py:407` records only on success.
- `src/elspeth/plugins/llm/azure_batch.py:423` calls `client.batches.create` without try/except.
- `src/elspeth/plugins/llm/azure_batch.py:485` calls `client.batches.retrieve` without try/except.
- `src/elspeth/plugins/llm/azure_batch.py:593` calls `client.files.content` without try/except.
- `CLAUDE.md:114` specifies external API calls must be wrapped.

## Impact

- User-facing impact: pipeline crashes on transient Azure errors with no controlled error routing.
- Data integrity / security impact: missing CallStatus.ERROR records break audit completeness.
- Performance or cost impact: retry logic cannot trigger; manual reruns may duplicate batches.

## Root Cause Hypothesis

- Missing try/except around Azure SDK calls means failures are neither recorded nor converted into TransformResult.error.

## Proposed Fix

- Code changes (modules/files):
  - Wrap `client.files.create`, `client.batches.create`, `client.batches.retrieve`, and `client.files.content` in try/except in `src/elspeth/plugins/llm/azure_batch.py`.
  - On exception, call `ctx.record_call` with CallStatus.ERROR and error details, then return TransformResult.error (or raise BatchPendingError for transient errors while preserving checkpoint).
- Config or schema changes: Unknown
- Tests to add/update:
  - Add tests in `tests/plugins/llm/test_azure_batch.py` that simulate SDK exceptions and assert error call recording plus error return behavior.
- Risks or migration steps:
  - Decide on retry semantics for transient errors to avoid premature batch failure.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:114`
- Observed divergence: external API calls are not wrapped; failures skip error recording.
- Reason (if known): no try/except around SDK calls.
- Alignment plan or decision needed: wrap external calls and record error outcomes.

## Acceptance Criteria

- SDK failures are recorded with CallStatus.ERROR and error details in the audit trail.
- The transform returns a controlled error or pending status instead of crashing.
- Batch checkpoints are preserved or cleared appropriately on failure.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_batch.py`
- New tests required: yes, simulate SDK exceptions and verify error call recording.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md:114`
