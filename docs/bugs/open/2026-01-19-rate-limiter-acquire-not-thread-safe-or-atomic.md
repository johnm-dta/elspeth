# Bug Report: RateLimiter.acquire() is not locked/atomic across multi-rate limiters (unlike try_acquire)

## Summary

- `RateLimiter.try_acquire()` is protected by `self._lock` and pre-checks all buckets to avoid partial token consumption across multiple rate windows.
- `RateLimiter.acquire()` is not locked and acquires sequentially across `self._limiters` without any atomicity guarantees.
- In multi-rate mode (per-second + per-minute) and/or with concurrency, this can lead to:
  - interleaving acquisitions across threads
  - partial consumption if a later limiter blocks/raises after an earlier limiter already consumed tokens
  - inconsistencies vs the intended “atomic across rate windows” contract that `try_acquire()` implements

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 3 (core infrastructure) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/core/rate_limit/limiter.py`

## Steps To Reproduce

One deterministic way to reproduce in a unit test is to force a later limiter acquisition to fail after an earlier acquisition succeeded:

1. Construct a `RateLimiter` with both `requests_per_second` and `requests_per_minute`.
2. Monkeypatch the underlying second limiter `try_acquire` to succeed and the minute limiter `try_acquire` to raise `BucketFullException` (or a timeout-derived exception).
3. Call `RateLimiter.acquire()`.
4. Observe that tokens were consumed in the first limiter despite overall acquisition failing.

(Concurrency repro)
1. Create a shared `RateLimiter` instance.
2. Call `acquire()` concurrently from multiple threads.
3. Observe interleavings / partial consumption become possible because `acquire()` does not take the instance lock.

## Expected Behavior

- `acquire()` should be thread-safe (like `try_acquire()`) and should not partially consume tokens across multi-rate windows if the overall acquisition cannot complete.

## Actual Behavior

- `acquire()` loops over limiters without using `self._lock`, so it can interleave across threads.
- Partial consumption is possible if later limiter acquisition fails after earlier limiter tokens were acquired.

## Evidence

- `acquire()` has no locking and acquires sequentially:
  - `src/elspeth/core/rate_limit/limiter.py:203-212`
- `try_acquire()` is explicitly locked and implements atomicity protection:
  - `src/elspeth/core/rate_limit/limiter.py:232-260`

## Impact

- User-facing impact: unexpected rate limiting behavior under concurrency; potential over-throttling (tokens “lost”) or inconsistent acquisition semantics.
- Data integrity / security impact: low (correctness/operational reliability issue).
- Performance or cost impact: potentially high if tokens are consumed unnecessarily and cause avoidable waits/timeouts.

## Root Cause Hypothesis

- Atomic multi-window logic was implemented for `try_acquire()` (because partial consumption is obvious there), but `acquire()` was left as a simple sequential wrapper and never updated to use the same lock/atomicity contract.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/rate_limit/limiter.py`:
    - Wrap `acquire()` in `with self._lock:` to prevent interleaving across threads.
    - Consider mirroring the atomic approach from `try_acquire()` for the blocking case:
      - check capacity across buckets, compute required wait, sleep once, then consume tokens for all limiters
      - or attempt acquisition in a loop that guarantees “all-or-nothing” as much as the underlying library allows
- Tests to add/update:
  - Add a unit test that forces a failure in the second limiter and asserts the first limiter did not retain consumed tokens.
  - Add a concurrency test that calls `acquire()` from multiple threads and asserts no exceptions and consistent bucket counts.
- Risks or migration steps:
  - Cross-process atomicity cannot be guaranteed with SQLite persistence, but per-instance thread safety should still be enforced.

## Architectural Deviations

- Spec or doc reference: N/A (operational correctness)
- Observed divergence: `try_acquire()` is atomic/thread-safe; `acquire()` is not.
- Reason (if known): incremental hardening for `try_acquire()` without porting changes to `acquire()`.
- Alignment plan or decision needed: define whether `acquire()` is intended to be thread-safe and atomic across windows; current code implies yes.

## Acceptance Criteria

- `acquire()` is locked and cannot interleave across threads within a single RateLimiter instance.
- Unit test demonstrates no partial consumption when later limiter acquisition fails.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/rate_limit/test_limiter.py`
- New tests required: yes (acquire atomicity + concurrency)

## Notes / Links

- Related issues/PRs: N/A
