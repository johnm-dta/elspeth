# Bug Report: Passthrough aggregation failure leaves buffered tokens non-terminal

## Summary

- When a passthrough aggregation flush returns an error, only the triggering token is marked FAILED. Previously buffered tokens remain in the BUFFERED outcome even though the batch is cleared, so they never reach a terminal state.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/engine/processor.py for bugs; create reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Configure a batch-aware transform with aggregation `output_mode: passthrough`.
2. Make the batch transform return `TransformResult.error(...)` on flush.
3. Feed multiple rows so at least one row is buffered before the flush.

## Expected Behavior

- All tokens in the failed batch transition to a terminal outcome (FAILED/QUARANTINED) or are retried explicitly.

## Actual Behavior

- Only the triggering token is marked FAILED; earlier buffered tokens stay at BUFFERED even though the batch buffer is cleared.

## Evidence

- `src/elspeth/engine/processor.py:202-222` records FAILED for only `current_token` on flush error.
- Passthrough buffering records BUFFERED outcomes for prior tokens (`src/elspeth/engine/processor.py:385-401`).
- `AggregationExecutor.execute_flush()` clears buffers on failure (`src/elspeth/engine/executors.py:1006-1017`), so buffered tokens will never be reprocessed.

## Impact

- User-facing impact: Rows disappear silently on batch failure.
- Data integrity / security impact: Violates “every token reaches terminal state”; audit trail shows stuck BUFFERED outcomes.
- Performance or cost impact: Manual remediation required; batch retry logic undermined.

## Root Cause Hypothesis

- Error handling in `_process_batch_aggregation_node()` only records failure for the triggering token and ignores buffered tokens.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add a test ensuring all buffered tokens receive terminal outcomes on flush error.
- Risks or migration steps: None; ensures audit completeness.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md “Every row reaches exactly one terminal state”.
- Observed divergence: Buffered tokens can be left non-terminal after a failed flush.
- Reason (if known): Flush error path only handles `current_token`.
- Alignment plan or decision needed: Define terminal outcome for failed batches (likely FAILED/QUARANTINED for all members).

## Acceptance Criteria

- All tokens in a failed passthrough batch receive terminal outcomes.
- No BUFFERED outcomes remain after a failed flush.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k aggregation_passthrough_failure`
- New tests required: Yes (batch failure terminal outcomes).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`
