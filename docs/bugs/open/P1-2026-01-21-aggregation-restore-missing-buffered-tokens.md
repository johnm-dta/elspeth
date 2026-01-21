# Bug Report: Aggregation checkpoint restore drops buffered tokens, flush crashes on resume

## Summary

- `AggregationExecutor.restore_from_checkpoint()` restores buffered rows but clears `_buffer_tokens` and never reconstructs `TokenInfo`.
- `execute_flush()` assumes `_buffer_tokens` is populated and unconditionally uses `buffered_tokens[0]`, so a resumed aggregation flush can crash with `IndexError` or lose token metadata needed for audit lineage.

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

- Goal or task prompt: deep dive into `src/elspeth/engine/executors.py` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/engine/executors.py`

## Steps To Reproduce

1. Create an `AggregationExecutor` and buffer at least one row for an aggregation node.
2. Call `get_checkpoint_state()` to capture the buffer state.
3. Create a new `AggregationExecutor` and call `restore_from_checkpoint()` with the captured state.
4. Call `execute_flush()` for that node.

## Expected Behavior

- Restored buffers include the original `TokenInfo` (or are reconstructed), allowing `execute_flush()` to complete and preserve audit lineage.

## Actual Behavior

- `_buffer_tokens` is empty after restore; `execute_flush()` tries to read `buffered_tokens[0]` and crashes, or proceeds without token metadata.

## Evidence

- `execute_flush()` assumes `buffered_tokens[0]` exists:
  - `src/elspeth/engine/executors.py:886`
  - `src/elspeth/engine/executors.py:896`
  - `src/elspeth/engine/executors.py:897`
- `restore_from_checkpoint()` clears `_buffer_tokens` and never reconstructs `TokenInfo`:
  - `src/elspeth/engine/executors.py:1085`
  - `src/elspeth/engine/executors.py:1099`
  - `src/elspeth/engine/executors.py:1102`

## Impact

- User-facing impact: resume/flush can crash for aggregations, breaking recovery flows.
- Data integrity / security impact: missing token metadata breaks audit lineage (batch membership → node_state linkage).
- Performance or cost impact: failed resumes and repeated runs.

## Root Cause Hypothesis

- Checkpoint state stores token IDs but `restore_from_checkpoint()` does not rehydrate `TokenInfo`, leaving `_buffer_tokens` empty.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`: reconstruct `TokenInfo` from stored token IDs (query tokens table or accept token metadata in checkpoint), or store full token metadata in checkpoint state.
  - Guard `execute_flush()` against empty `_buffer_tokens` when `buffered_rows` exist (fail with a clear error and remediation).
- Config or schema changes: none.
- Tests to add/update:
  - Add a recovery test that buffers rows, checkpoints, restores, and flushes without error.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/requirements.md` (aggregation checkpoint/restore).
- Observed divergence: restored aggregation buffers are not executable.
- Reason (if known): restore path only restores rows, not token metadata.
- Alignment plan or decision needed: define how token metadata is persisted/restored for aggregation checkpoints.

## Acceptance Criteria

- Resumed aggregations can flush successfully with correct token lineage.
- `execute_flush()` no longer crashes on restored buffers.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -k aggregation_resume`
- New tests required: yes (aggregation checkpoint restore + flush).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
