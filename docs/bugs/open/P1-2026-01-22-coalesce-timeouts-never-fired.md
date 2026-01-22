# Bug Report: Coalesce timeouts are never checked (check_timeouts is unused)

## Summary

- `CoalesceExecutor.check_timeouts()` is defined but never invoked, so `best_effort` and `quorum` timeouts do not fire; coalesce groups wait until end-of-source (or forever in streaming sources).

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
- Notable tool calls or steps: code inspection of coalesce executor and orchestrator; rg search for check_timeouts usage

## Steps To Reproduce

1. Configure a pipeline with a fork and a coalesce set to `policy: best_effort` (or `quorum`) and a short `timeout_seconds`.
2. Use a long-running or streaming source so end-of-source is not reached quickly.
3. Run the pipeline and wait past the timeout.

## Expected Behavior

- After `timeout_seconds`, the coalesce should merge (best_effort) or resolve (quorum) without waiting for end-of-source.

## Actual Behavior

- No timeout-driven merge occurs; pending coalesces only resolve on full arrival or end-of-source flush.

## Evidence

- Timeout handler exists but has no callers: `src/elspeth/engine/coalesce_executor.py:303`
- Comment claims a timeout loop exists, but no loop calls it: `src/elspeth/engine/coalesce_executor.py:112`
- Orchestrator only flushes at end-of-source: `src/elspeth/engine/orchestrator.py:866`

## Impact

- User-facing impact: pipelines with timeouts hang or emit delayed results.
- Data integrity / security impact: none directly, but audit timing data is wrong.
- Performance or cost impact: unbounded pending state growth on long-running sources.

## Root Cause Hypothesis

- `check_timeouts()` was implemented but never wired into orchestrator/processor scheduling.

## Proposed Fix

- Code changes (modules/files):
  - Call `CoalesceExecutor.check_timeouts()` periodically during processing (e.g., per row or on a timer) and enqueue any merged tokens.
  - Ensure timeout-driven failures are handled for policies that require it.
- Config or schema changes: none.
- Tests to add/update:
  - Add an integration test with `best_effort` timeout verifying merge without end-of-source.
- Risks or migration steps:
  - Ensure timeout checks do not introduce excessive overhead; consider batching or interval checks.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1092`
- Observed divergence: timeouts are defined but never triggered.
- Reason (if known): missing orchestration loop wiring.
- Alignment plan or decision needed: implement periodic timeout checks in execution loop.

## Acceptance Criteria

- A coalesce with `best_effort` or `quorum` timeout resolves when timeout elapses during streaming runs.
- No pending coalesce remains past its timeout unless policy explicitly allows it.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k timeout`
- New tests required: yes (timeout merge in streaming mode)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
