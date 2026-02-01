# Bug Report: Pre-check uses bucket.count(), causing false rate-limit rejections

## Summary

- `_would_all_buckets_accept()` uses `bucket.count()` which returns total retained items without window filtering, causing stale counts to reject valid requests.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/rate_limit/limiter.py:223-239` - uses `bucket.count()` for pre-check
- pyrate-limiter's leak interval is 10s by default
- A 1/sec limit can effectively become 1/10sec due to stale counts

## Impact

- User-facing impact: Rate limiting stricter than configured
- Data integrity: None

## Proposed Fix

- Use time-window-aware counting or remove pre-check in favor of letting pyrate-limiter handle rejection

## Acceptance Criteria

- Rate limiting matches configured limits, not stricter

## Verification (2026-02-01)

**Status: OBE**

- The `_would_all_buckets_accept()` pre-check and any `bucket.count()` usage are no longer present in the current rate limiter implementation. (`src/elspeth/core/rate_limit/limiter.py` has no such function or count usage.)

## Closure Report (2026-02-01)

**Status:** CLOSED (OBE)

### Closure Notes

- Pre-check logic was removed; current limiter relies on `Limiter.try_acquire()` directly.
