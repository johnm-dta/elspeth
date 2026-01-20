# Bug Report: Orchestrator.resume() is partial and returns misleading RunResult (does not process unprocessed rows)

## Summary

- `Orchestrator.resume()` updates run status and retries incomplete batches, but it explicitly does not continue processing unprocessed rows (TODO) and returns a `RunResult` with zeros and `status=RUNNING`.
- This can mislead callers/operators into believing recovery resumed work, when it only performed partial bookkeeping.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: checkpointing + recovery enabled
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `resume()` implementation

## Steps To Reproduce

1. Create a failed run with a checkpoint (e.g., crash mid-run).
2. Call `RecoveryManager.get_resume_point(run_id)` and then `Orchestrator.resume(...)`.

## Expected Behavior

- Either:
  - resume processes remaining rows and returns an accurate `RunResult`, or
  - resume fails fast with a clear “not implemented” error instead of returning a misleading partial result.

## Actual Behavior

- resume returns a `RunResult` with zero counts and `status=RUNNING` but does not process remaining rows.

## Evidence

- TODO and placeholder return:
  - `src/elspeth/engine/orchestrator.py:897`
  - `src/elspeth/engine/orchestrator.py:904`

## Impact

- User-facing impact: operators can believe recovery succeeded when it didn’t actually resume processing.
- Data integrity / security impact: incomplete recovery can leave tokens/batches in ambiguous states without clear reporting.
- Performance or cost impact: wasted operator time; repeated manual interventions.

## Root Cause Hypothesis

- Recovery implementation was started with batch retry bookkeeping but row processing resume logic was deferred.

## Proposed Fix

- Code changes (modules/files):
  - If recovery is not ready: raise `NotImplementedError` after `_handle_incomplete_batches(...)` (or return a distinct status indicating partial recovery).
  - Otherwise: implement the planned steps in the TODO:
    - query `RecoveryManager.get_unprocessed_rows(run_id)`
    - restore aggregation state into `RowProcessor`
    - reprocess rows and write to sinks with correct checkpoint semantics
- Tests to add/update:
  - Add end-to-end recovery test that fails mid-run and successfully resumes all remaining rows.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (audit/recovery integrity expectations)
- Observed divergence: recovery surface exists but does not complete recovery.
- Alignment plan or decision needed: decide whether to expose partial recovery API or hide behind feature flag.

## Acceptance Criteria

- resume either completes row processing with correct counts or fails fast with a clear “not implemented” message.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator_recovery.py`
- New tests required: yes (resume actually processes rows)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`
