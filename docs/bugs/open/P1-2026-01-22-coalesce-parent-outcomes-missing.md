# Bug Report: Coalesce never records COALESCED outcomes for parent tokens

## Summary

- Coalesce merges branch tokens but never records a terminal `RowOutcome.COALESCED` for the consumed parent tokens; only the merged token gets an outcome record.
- This violates the audit contract that every token reaches exactly one terminal state and that child tokens are marked COALESCED.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (fix/rc1-bug-burndown-session-2)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into coalesce_executor, identify bugs, create bug docs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of coalesce executor and processor

## Steps To Reproduce

1. Configure a pipeline with a fork followed by a coalesce (any policy).
2. Run a row through both branches so the coalesce merges.
3. Inspect `token_outcomes` for the branch tokens.

## Expected Behavior

- Each branch token should be recorded with terminal outcome `COALESCED` (and join group metadata) when merged.

## Actual Behavior

- Branch tokens have no terminal outcome recorded; only the merged token is recorded as `COALESCED`.

## Evidence

- Coalesce merge only records node states, not token outcomes, for consumed tokens: `src/elspeth/engine/coalesce_executor.py:236`
- Processor records `COALESCED` outcome for the merged token instead: `src/elspeth/engine/processor.py:983`
- Contract requires child tokens to be marked `COALESCED`: `docs/contracts/plugin-protocol.md#L1111`

## Impact

- User-facing impact: explain/replay shows branch tokens with missing terminal state; audit trail is incomplete.
- Data integrity / security impact: violates AUD-001 terminal state guarantee for every token.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Coalesce implementation focuses on node states and merged token creation, but does not emit `record_token_outcome()` for parent tokens.

## Proposed Fix

- Code changes (modules/files):
  - In `CoalesceExecutor._execute_merge()`, record `RowOutcome.COALESCED` for each consumed token.
  - Expose or retrieve the `join_group_id` created by `LandscapeRecorder.coalesce_tokens()` so it can be stored with the outcome.
  - Consider whether the merged token should have a different outcome (e.g., `COMPLETED` or `ROUTED`) to avoid double-terminal labeling.
- Config or schema changes: none.
- Tests to add/update:
  - Add a coalesce integration test asserting `token_outcomes` contains COALESCED for each branch token.
- Risks or migration steps:
  - Ensure outcome uniqueness constraints are respected (terminal outcomes are unique per token).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1111`
- Observed divergence: consumed child tokens are not marked as COALESCED.
- Reason (if known): outcome recording exists in processor for merged token, but not for parents.
- Alignment plan or decision needed: decide correct outcome for merged token vs parents and update recording logic.

## Acceptance Criteria

- After a merge, every consumed branch token has a COALESCED outcome recorded with join group metadata.
- Merged token continues and receives the appropriate terminal outcome later in the pipeline.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor_outcomes.py -k coalesce`
- New tests required: yes (coalesce parent outcome recording)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
