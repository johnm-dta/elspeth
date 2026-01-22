# Bug Report: Duplicate branch arrivals overwrite earlier tokens without error

## Summary

- When a second token arrives for the same `(row_id, branch_name)`, the executor overwrites the first token in `pending.arrived` without error.
- This silently drops data and can merge the wrong token if duplicates occur due to retries or bugs.

## Severity

- Severity: major
- Priority: P2

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

1. Configure a coalesce with branches `A` and `B`.
2. Cause branch `A` to emit two tokens for the same `row_id` (e.g., via retry bug or duplicate processing).
3. Observe that only the last token is used in the merge.

## Expected Behavior

- Duplicate arrivals for the same branch should raise an error or be explicitly rejected with a recorded failure.

## Actual Behavior

- The later token silently overwrites the earlier one, losing data and masking upstream bugs.

## Evidence

- Arrival overwrites per-branch token without validation: `src/elspeth/engine/coalesce_executor.py:172`

## Impact

- User-facing impact: merged output can be inconsistent or derived from the wrong token.
- Data integrity / security impact: silent data loss and audit gaps.
- Performance or cost impact: none directly, but debugging is harder.

## Root Cause Hypothesis

- `pending.arrived` is treated as a simple map with no duplicate detection, so later arrivals replace earlier ones.

## Proposed Fix

- Code changes (modules/files):
  - Detect duplicate arrivals for the same branch and raise a hard error (bug in engine) or record a failure outcome.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that duplicate branch arrivals raise or are recorded as failures.
- Risks or migration steps:
  - If duplicates can occur legitimately, define explicit de-duplication semantics and document them.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1109`
- Observed divergence: duplicate branch arrivals are silently overwritten.
- Reason (if known): no explicit duplicate handling.
- Alignment plan or decision needed: enforce one token per branch per row_id.

## Acceptance Criteria

- Duplicate arrivals for the same branch are detected and handled deterministically (error or explicit failure record).
- The merge uses exactly one token per branch per row_id.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k duplicate`
- New tests required: yes (duplicate branch arrival detection)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
