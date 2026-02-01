# P2-2026-01-29: Dispatch Gate Bypass on Retry

| Field | Value |
|-------|-------|
| **ID** | P2-2026-01-29-dispatch-gate-bypass-on-retry |
| **Priority** | P2 - High |
| **Component** | plugins/pooling/executor.py |
| **Status** | CLOSED - FIXED |
| **Discovered** | 2026-01-29 (code review) |
| **Fixed** | 2026-01-29 |

## Summary

Workers that hit capacity errors could bypass the global dispatch gate after retry, allowing bursty dispatches that violate `min_dispatch_delay_ms` pacing guarantees.

## Root Cause

The executor used a `just_retried` flag to skip `_wait_for_dispatch_gate()` after capacity error retries. The rationale was "avoid double-sleeping" since the worker already slept during retry backoff.

This reasoning was flawed because:
1. **Retry backoff** = personal cooldown for THIS worker after hitting capacity
2. **Dispatch gate** = global coordination ensuring ALL workers maintain minimum spacing

While Worker A sleeps for backoff, Workers B, C, D may continue dispatching. When A wakes, it must still check the gate because D might have just dispatched.

## Reproduction Scenario

```
Timeline with min_dispatch_delay_ms = 100ms:
T=0:    Worker A dispatches → capacity error, starts 500ms backoff
T=100:  Worker B dispatches (gate OK: 100ms since last)
T=200:  Worker C dispatches (gate OK: 100ms since last)
T=490:  Worker D dispatches (gate OK: 290ms since last)
T=500:  Worker A wakes, skips gate check, dispatches immediately
        ❌ VIOLATION: only 10ms since Worker D's dispatch!
```

## Impact

- Short-interval bursts during capacity-error scenarios
- Potential for increased rate limiting from API providers
- `min_dispatch_delay_ms` guarantee violated

## Fix

Removed the `just_retried` optimization. Workers now always check the dispatch gate after retry.

```python
# Before (buggy)
if not just_retried:
    self._wait_for_dispatch_gate()
just_retried = False

# After (fixed)
# Always check the gate - retry backoff is personal, gate is global
self._wait_for_dispatch_gate()
```

## Regression Test

Added `TestDispatchGateAfterRetry::test_retry_respects_dispatch_gate_timing` which verifies all consecutive dispatches (including post-retry) respect `min_dispatch_delay_ms`.

## Files Changed

- `src/elspeth/plugins/pooling/executor.py` - Removed `just_retried` bypass
- `tests/plugins/pooling/test_executor_retryable_errors.py` - Added regression test
