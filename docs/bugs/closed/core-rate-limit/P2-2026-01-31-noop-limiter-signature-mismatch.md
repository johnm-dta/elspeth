# Bug Report: NoOpLimiter.acquire() signature does not match RateLimiter.acquire()

## Summary

- `NoOpLimiter.acquire()` lacks `timeout` parameter that `RateLimiter.acquire()` has, causing TypeError when rate limiting is disabled and caller passes timeout.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/rate_limit/registry.py:22-27` - `NoOpLimiter.acquire(self, weight: int = 1)` lacks timeout.
- `src/elspeth/core/rate_limit/limiter.py:170-193` - `RateLimiter.acquire(self, weight: int = 1, timeout: float | None = None)` accepts timeout.

## Impact

- User-facing impact: TypeError when switching between rate-limited and non-rate-limited configs
- Data integrity: None

## Proposed Fix

- Add `timeout: float | None = None` parameter to `NoOpLimiter.acquire()`

## Acceptance Criteria

- NoOpLimiter and RateLimiter have matching signatures

## Resolution (2026-02-02)

**Status: FIXED**

### Root Cause

When `timeout` parameter was added to `RateLimiter.acquire()`, the corresponding change to `NoOpLimiter.acquire()` was missed. Since `RateLimitRegistry.get_limiter()` returns `RateLimiter | NoOpLimiter`, both classes must have identical method signatures for callers to use them interchangeably.

### Fix Applied

1. Added `timeout: float | None = None` parameter to `NoOpLimiter.acquire()` in `registry.py:22`
2. Added regression test `test_acquire_accepts_timeout_parameter` in `tests/core/rate_limit/test_registry.py`

### Files Changed

- `src/elspeth/core/rate_limit/registry.py` - Added timeout parameter to `NoOpLimiter.acquire()`
- `tests/core/rate_limit/test_registry.py` - Added regression test

### Verification

- All 51 rate_limit tests pass
- mypy type checking passes
