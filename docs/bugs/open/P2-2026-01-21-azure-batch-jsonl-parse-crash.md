# Bug Report: Azure batch crashes on malformed JSONL output

## Summary

- AzureBatchLLMTransform assumes every output line is valid JSON with a `custom_id`. Any malformed line or missing key raises JSONDecodeError/KeyError and crashes the transform instead of returning per-row errors.

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
- Data set or fixture: any batch run where output JSONL contains malformed line

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Run `azure_batch_llm` and simulate a partial/garbled JSONL output (e.g., truncate the output file).
2. Resume the batch so `_download_results` runs.
3. Observe JSONDecodeError or KeyError during parsing.

## Expected Behavior

- Malformed output lines are handled as external data errors, resulting in TransformResult.error or per-row error markers without crashing the pipeline.

## Actual Behavior

- json.loads and direct `result["custom_id"]` access raise exceptions and crash the transform.

## Evidence

- Unchecked json.loads and `custom_id` indexing at `src/elspeth/plugins/llm/azure_batch.py:603` and `src/elspeth/plugins/llm/azure_batch.py:608`.

## Impact

- User-facing impact: batch runs can fail during completion even if most rows succeeded.
- Data integrity / security impact: no audit record for malformed outputs.
- Performance or cost impact: reruns required.

## Root Cause Hypothesis

- Output parsing treats external data as trusted and lacks error handling.

## Proposed Fix

- Code changes (modules/files): wrap json.loads and key access in try/except; emit per-row errors or TransformResult.error with context.
- Config or schema changes: N/A
- Tests to add/update:
  - Add tests with malformed JSONL lines and missing custom_id.
- Risks or migration steps:
  - Ensure partial success semantics remain intact.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md external system boundaries must be wrapped.
- Observed divergence: external output parsing can crash.
- Reason (if known): missing guardrails.
- Alignment plan or decision needed: standardize error handling for batch output parsing.

## Acceptance Criteria

- Malformed JSONL output yields structured errors and no unhandled exceptions.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_batch.py -v`
- New tests required: yes, malformed output handling.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard
