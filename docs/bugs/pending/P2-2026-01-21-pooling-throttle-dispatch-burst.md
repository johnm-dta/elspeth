# Bug Report: Pooled dispatch delay is per-worker (bursty dispatches)

## Summary

- AIMD throttle delay is applied inside each worker thread, so with pool_size > 1 multiple requests dispatch at the same time after a shared sleep. This violates the design requirement of a global delay between dispatches and causes bursty traffic that can trigger rate-limit spikes.

## Severity

- Severity: major
- Priority: P2

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

- Goal or task prompt: Deep dive src/elspeth/plugins/pooling for bugs; create bug reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Configure pooling with `pool_size=4` and `min_dispatch_delay_ms=500` (or any non-zero delay).
2. Run `execute_batch` with 4 rows and a `process_fn` that records `time.perf_counter()` at function entry.
3. Compare the recorded start times for each row.

## Expected Behavior

- Dispatches should be globally spaced so each request starts at least `current_delay_ms` after the previous dispatch, per design.

## Actual Behavior

- All workers sleep concurrently and then dispatch nearly simultaneously, creating bursts that violate the intended inter-dispatch delay.

## Evidence

- Code: `src/elspeth/plugins/pooling/executor.py:256-265` applies delay per worker.
- Spec: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:83-85` requires a dispatcher that waits `current_delay` between dispatches.

## Impact

- User-facing impact: Higher chance of repeated 429/503s under rate limits.
- Data integrity / security impact: None directly, but audit trail will show more capacity retries than necessary.
- Performance or cost impact: Increased backoff cycles and wasted retries.

## Root Cause Hypothesis

- Throttle delay is implemented inside each worker instead of a shared dispatcher or global pacing gate.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/plugins/pooling/executor.py`
- Config or schema changes: None
- Tests to add/update: Add a test that asserts dispatch start times are spaced by at least `current_delay_ms` across the pool.
- Risks or migration steps: Ensure new pacing does not reintroduce deadlocks; avoid holding semaphore during global delay.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:83-85`
- Observed divergence: No centralized dispatcher; per-thread delays allow burst dispatching.
- Reason (if known): Implementation simplified to worker-local sleep.
- Alignment plan or decision needed: Implement a shared pacing gate or dispatcher as specified.

## Acceptance Criteria

- Dispatch timestamps across pool_size workers are globally spaced by `current_delay_ms` under steady state.
- New test passes and existing pooling tests remain green.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_pooled_executor.py -k dispatch`
- New tests required: Yes (global pacing test).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md`
