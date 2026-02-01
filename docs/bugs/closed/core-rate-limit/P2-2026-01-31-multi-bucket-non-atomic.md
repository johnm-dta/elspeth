# Bug Report: Multi-bucket acquisition is non-atomic across processes

## Summary

- Sequential bucket acquisition can consume tokens from earlier buckets before failing on later buckets. Cross-process rate limiting lacks atomicity.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/rate_limit/limiter.py:251-270` - per-instance lock only, no cross-process atomicity
- Sequential bucket acquisition means partial token consumption on failure

## Impact

- User-facing impact: Token leakage in multi-process setups
- Data integrity: None, but rate limits may be less effective

## Proposed Fix

- Document as known limitation, or implement two-phase locking

## Acceptance Criteria

- Behavior documented, or atomicity guaranteed across processes

## Verification (2026-02-01)

**Status: OBE**

- The current `RateLimiter` implementation only configures a **single** rate bucket (`rates = [Rate(requests_per_minute, ...)]`), so multi-bucket acquisition no longer exists in this code path. (`src/elspeth/core/rate_limit/limiter.py:151-168`)

## Closure Report (2026-02-01)

**Status:** CLOSED (OBE)

### Closure Notes

- Multi-bucket acquisition is no longer present in the current rate limiter implementation.
