# Bug Report: AzureBlobSource aborts on invalid JSONL lines instead of quarantining

## Summary

- `AzureBlobSource._load_jsonl()` raises `ValueError` on any malformed JSON line, aborting the entire source load.
- This violates the Tier-3 trust boundary behavior and skips `ctx.record_validation_error`, so parse errors are not captured in the audit trail.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/azure` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/plugins/azure/blob_source.py`

## Steps To Reproduce

1. Upload a JSONL blob containing at least one malformed line (e.g., truncated JSON on line 2).
2. Configure `AzureBlobSource` with `format: jsonl` and a valid schema.
3. Run a pipeline that reads from the blob.

## Expected Behavior

- Malformed lines are quarantined (with `ctx.record_validation_error`), and valid lines continue to process.

## Actual Behavior

- The first malformed line raises `ValueError`, aborting the source load and the run.

## Evidence

- JSONL parse errors are re-raised as `ValueError` with no quarantine path:
  - `src/elspeth/plugins/azure/blob_source.py:433`
  - `src/elspeth/plugins/azure/blob_source.py:438`
  - `src/elspeth/plugins/azure/blob_source.py:440`
  - `src/elspeth/plugins/azure/blob_source.py:442`

## Impact

- User-facing impact: a single bad JSONL line halts the entire run.
- Data integrity / security impact: parse failures are not recorded in `validation_errors`, weakening auditability.
- Performance or cost impact: repeated runs needed to isolate bad data.

## Root Cause Hypothesis

- `AzureBlobSource._load_jsonl()` implements parse errors as fatal exceptions instead of quarantining lines like `JSONSource` does.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/azure/blob_source.py`: mirror `JSONSource._load_jsonl()` behavior by catching `JSONDecodeError`, recording a validation error with raw line + line number, yielding `SourceRow.quarantined`, and continuing.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that injects an invalid JSONL line and asserts the run continues with a quarantined row and audit entry.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md` (Three-Tier Trust Model), `src/elspeth/plugins/sources/json_source.py` JSONL handling.
- Observed divergence: Azure JSONL parsing crashes instead of quarantining invalid external data.
- Reason (if known): Azure implementation did not reuse JSONSource parsing logic.
- Alignment plan or decision needed: align source parsing behavior across local and Azure JSONL sources.

## Acceptance Criteria

- Invalid JSONL lines are quarantined with audit entries and do not abort the run.
- Valid lines after malformed ones still produce `SourceRow.valid()` output.

## Tests

- Suggested tests to run:
  - `pytest tests/plugins/azure/test_blob_source.py -k jsonl`
- New tests required: yes (JSONL parse-error quarantine)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
