# Bug Report: require_all coalesce ignores timeout_seconds and never resolves until end-of-source

## Summary

- `CoalesceExecutor.check_timeouts()` never handles `policy="require_all"`, so `timeout_seconds` is silently ignored and pending coalesces persist indefinitely until `flush_pending()`.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-30
- Related run/issue ID: N/A

## Environment

- Commit/branch: Unknown
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Fork/coalesce pipeline with `policy=require_all` and `timeout_seconds` set; one branch delayed or missing

## Agent Context (if relevant)

- Goal or task prompt: static analysis deep bug audit of `coalesce_executor.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a coalesce with `policy: require_all` and a small `timeout_seconds` (e.g., 1s).
2. Run a fork where only one branch reaches the coalesce (the other branch fails or stalls).
3. Wait beyond `timeout_seconds` and inspect audit/pipeline state.

## Expected Behavior

- After `timeout_seconds`, the pending coalesce should resolve as a failure (e.g., `incomplete_branches`) with node states and token outcomes recorded, and the pending entry cleared.

## Actual Behavior

- `check_timeouts()` never triggers for `require_all`, so the pending entry persists with no failure outcome until end-of-source (or indefinitely for streaming sources).

## Evidence

- `src/elspeth/engine/coalesce_executor.py:469-582` — `check_timeouts()` handles `best_effort` and `quorum` only; there is no `require_all` branch.
- `src/elspeth/engine/coalesce_executor.py:584-719` — `require_all` failure handling exists only in `flush_pending()` (end-of-source).
- `docs/reference/configuration.md:420` — `timeout_seconds` described as “Max wait time” without policy restriction.
- `src/elspeth/core/config.py:375-400` — `timeout_seconds` is allowed for `require_all` (no validator rejecting it).

## Impact

- User-facing impact: Pipelines with `require_all` + timeout can hang indefinitely when a branch never arrives (especially streaming sources).
- Data integrity / security impact: Tokens remain in non-terminal state; audit trail lacks failure resolution until end-of-source, violating “every row reaches a terminal state.”
- Performance or cost impact: Pending entries accumulate, increasing memory usage and potentially stalling progress.

## Root Cause Hypothesis

- `check_timeouts()` omits handling for `policy="require_all"`, so timeouts configured for require_all are ignored.

## Proposed Fix

- Code changes (modules/files):
  - Add `require_all` timeout handling in `src/elspeth/engine/coalesce_executor.py` `check_timeouts()` mirroring the `flush_pending()` failure path (`incomplete_branches`), including node state completion and `record_token_outcome()` for arrived tokens.
- Config or schema changes: None required if runtime behavior is added.
- Tests to add/update:
  - Add unit test for `require_all` timeout in `tests/engine/test_coalesce_executor.py` to assert failure is recorded and pending cleared.
  - Add integration test for streaming/long-running scenario to ensure timeout fires without end-of-source.
- Risks or migration steps:
  - Existing configs relying on “wait forever” behavior will now fail at timeout (expected per `timeout_seconds`).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/reference/configuration.md:417` (“timeout_seconds: Max wait time”)
- Observed divergence: `timeout_seconds` has no effect for `require_all` policy during runtime timeouts.
- Reason (if known): Missing branch in `check_timeouts()` for `require_all`.
- Alignment plan or decision needed: Define `require_all` timeout semantics explicitly (fail with `incomplete_branches`) and enforce in executor.

## Acceptance Criteria

- A `require_all` coalesce with `timeout_seconds` reliably resolves to failure after timeout without waiting for end-of-source.
- Audit trail shows failed node states and token outcomes for arrived branches.
- No pending coalesce remains after timeout expiration.

## Verification (2026-02-01)

**Status: FIXED**

## Fix Implementation

**Changes made:**

1. **Added `require_all` handling in `check_timeouts()`** (`src/elspeth/engine/coalesce_executor.py`):
   - Added `elif settings.policy == "require_all":` branch after the `quorum` handling
   - Mirrors the failure logic from `flush_pending()` (lines 690-739)
   - Records `incomplete_branches` failure reason
   - Completes pending node states with FAILED status
   - Records FAILED token outcomes with error_hash
   - Cleans up pending entry and marks as completed
   - Returns CoalesceOutcome with failure metadata

2. **Added test `test_check_timeouts_records_failure_for_require_all`**:
   - Creates require_all coalesce with timeout_seconds=0.1
   - Accepts only 2 of 3 required branches
   - Advances clock past timeout
   - Verifies check_timeouts() returns failure outcome
   - Verifies FAILED outcomes recorded in audit trail
   - Verifies pending entry cleaned up
   - Verifies coalesce_metadata includes expected/arrived branches

**Files modified:**
- `src/elspeth/engine/coalesce_executor.py` - Added require_all timeout handling
- `tests/engine/test_coalesce_executor.py` - Added timeout test

**Test results:**
- 23 coalesce executor unit tests pass
- 19 coalesce integration tests pass
- mypy: no issues
- ruff: all checks pass

## Closure

- **Closed by:** Claude (systematic debugging fix)
- **Closure date:** 2026-02-01
- **Resolution:** Added require_all branch to check_timeouts() mirroring flush_pending() logic
