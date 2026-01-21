# Bug Report: Coalesce allows late arrivals to start a second merge (duplicate outputs)

## Summary

- After a merge completes, the executor deletes pending state but never marks the `(coalesce_name, row_id)` as closed, so late-arriving branch tokens create a new pending entry and can trigger a second merge.
- The `first` policy effectively merges every branch arrival because it never discards later branches.

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
- Notable tool calls or steps: code inspection of coalesce executor

## Steps To Reproduce

1. Configure a coalesce with `policy: first` and branches `A`, `B`.
2. Send branch `A` token first; coalesce merges immediately.
3. Send branch `B` token later for the same `row_id`.

## Expected Behavior

- After the first merge, later arrivals for the same `(coalesce_name, row_id)` should be discarded or explicitly recorded as late/ignored, not merged again.

## Actual Behavior

- A new pending entry is created and a second merge can occur, producing duplicate outputs for the same source row.

## Evidence

- Pending state is created whenever the key is missing: `src/elspeth/engine/coalesce_executor.py:159`
- Pending state is deleted after merge, with no closed-set tracking: `src/elspeth/engine/coalesce_executor.py:267`
- `first` policy merges on any single arrival: `src/elspeth/engine/coalesce_executor.py:201`
- Policy contract requires `first` to discard later arrivals: `docs/contracts/plugin-protocol.md#L1097`

## Impact

- User-facing impact: duplicate outputs and inconsistent downstream results for a single input row.
- Data integrity / security impact: audit trail can show multiple merged tokens for the same branch group.
- Performance or cost impact: extra merges and sink writes.

## Root Cause Hypothesis

- Coalesce executor tracks only pending merges and does not track completed merges by key.

## Proposed Fix

- Code changes (modules/files):
  - Track completed `(coalesce_name, row_id)` keys and reject or quarantine late arrivals.
  - For `first` policy, explicitly discard later arrivals after the first merge.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that sends a late arrival after a merge and asserts it is ignored or flagged.
- Risks or migration steps:
  - Define behavior for late arrivals (drop, error, or quarantine) and document it.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1097`
- Observed divergence: late arrivals are merged again instead of discarded.
- Reason (if known): no completed-set tracking.
- Alignment plan or decision needed: define and enforce late-arrival handling.

## Acceptance Criteria

- A `(coalesce_name, row_id)` can only be merged once.
- Late arrivals do not create additional merged tokens.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k first`
- New tests required: yes (late arrival suppression)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
