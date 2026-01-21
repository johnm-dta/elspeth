# Bug Report: expand_token does not deep copy row data, allowing cross-token mutation

## Summary

- `TokenManager.fork_token()` deep-copies row data to prevent branch mutations from leaking across siblings.
- `TokenManager.expand_token()` returns row data as-is, so expanded tokens can share mutable structures if a transform returns shared objects.
- This can cause downstream mutations in one expanded token to affect siblings, corrupting audit data.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/tokens.py` and create bug tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of token manager

## Steps To Reproduce

1. Implement a transform that returns `TransformResult.success_multi()` with shared objects, e.g. `rows = [row_template] * 2`.
2. Add a downstream transform that mutates `row_data` in place (e.g., `row["nested"]["x"] = 1`).
3. Run the pipeline and inspect both expanded tokens.
4. Observe that mutations applied to one token appear in the other.

## Expected Behavior

- Each expanded token should hold an isolated copy of its row data, preventing cross-token mutation.

## Actual Behavior

- Expanded tokens may share mutable structures, causing unintended data coupling between tokens.

## Evidence

- Forking uses deep copy to avoid shared state: `src/elspeth/engine/tokens.py` (`fork_token`).
- Expansion does not copy row data: `src/elspeth/engine/tokens.py` (`expand_token`).

## Impact

- User-facing impact: inconsistent or incorrect output rows when downstream transforms mutate data.
- Data integrity / security impact: audit trail can no longer reliably explain which row produced which output.
- Performance or cost impact: debugging and reprocessing costs increase.

## Root Cause Hypothesis

- Expansion path skipped the deep-copy protection that exists in fork_token.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/tokens.py`: deep copy each expanded row when constructing `TokenInfo`, mirroring fork_token behavior.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that expands a token with shared nested objects and asserts that mutations in one child do not affect others.
- Risks or migration steps:
  - Deep copy increases memory usage for large rows; ensure acceptable performance or provide a documented opt-out if needed.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): audit integrity principles in `CLAUDE.md` (no silent data corruption).
- Observed divergence: expanded tokens can share mutable data, unlike forked tokens.
- Reason (if known): expand_token omitted deep copy.
- Alignment plan or decision needed: decide whether expansion should guarantee data isolation like fork.

## Acceptance Criteria

- Expanded tokens never share mutable row_data structures.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_tokens.py -k expand`
- New tests required: yes (expand_token data isolation)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
