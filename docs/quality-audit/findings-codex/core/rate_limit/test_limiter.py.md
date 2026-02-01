# Test Defect Report

## Summary

- Tests assert wall‑clock timing thresholds around rate limiting, which makes them slow and can be flaky under CI scheduling variance.

## Severity

- Severity: major
- Priority: P1

## Category

- Sleepy Assertions

## Evidence

- `tests/core/rate_limit/test_limiter.py:145-171` measures real elapsed time around `acquire()` and asserts tight thresholds.
```python
start = time.monotonic()
limiter.acquire()
elapsed = time.monotonic() - start

assert elapsed < 0.1  # Should be near-instant
...
start = time.monotonic()
limiter.acquire()
elapsed = time.monotonic() - start

assert elapsed >= 0.9  # Should have waited ~1s
```
- `tests/core/rate_limit/test_limiter.py:652-659` relies on a 0.1s timeout; if the second call is delayed by scheduler jitter, it can succeed before the deadline.
```python
with pytest.raises(TimeoutError, match=r"Failed to acquire.*timeout"):
    limiter.acquire(weight=1, timeout=0.1)
```

## Impact

- Flaky failures on slower or loaded CI nodes; unnecessary test runtime from real waiting; timing regressions can be masked or misattributed to environment noise.

## Root Cause Hypothesis

- Blocking behavior was validated with real time because the limiter uses `time.sleep` internally and no fake clock is injected in tests.

## Recommended Fix

- Use a controlled fake clock in tests by monkeypatching `elspeth.core.rate_limit.limiter.time.monotonic` and `.sleep` so time advances deterministically without real waits.
- Drive time forward in the test to assert blocking/timeout behavior without wall‑clock dependency.
```python
def test_acquire_blocks_when_exceeded(monkeypatch):
    import elspeth.core.rate_limit.limiter as limiter_mod

    now = {"t": 0.0}

    def fake_monotonic():
        return now["t"]

    def fake_sleep(dt):
        now["t"] += dt

    monkeypatch.setattr(limiter_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(limiter_mod.time, "sleep", fake_sleep)
    ...
```
---
# Test Defect Report

## Summary

- Tests reach into private implementation details (`_requests_per_minute`, `_requests_per_second`, `_limiters`, `_buckets`) instead of asserting public behavior, coupling them to internals.

## Severity

- Severity: minor
- Priority: P2

## Category

- Infrastructure Gaps

## Evidence

- `tests/core/rate_limit/test_limiter.py:96-97` asserts a private field.
```python
with RateLimiter(name="test", requests_per_second=10, requests_per_minute=None) as limiter:
    assert limiter._requests_per_minute is None
```
- `tests/core/rate_limit/test_limiter.py:297-301` inspects private configuration.
```python
assert isinstance(default_limiter, RateLimiter)
assert isinstance(slow_limiter, RateLimiter)
assert default_limiter._requests_per_second == 10
assert slow_limiter._requests_per_second == 1
```
- `tests/core/rate_limit/test_limiter.py:371-373` checks internal registry storage.
```python
assert len(registry._limiters) == 0
```
- `tests/core/rate_limit/test_limiter.py:128-129`/`tests/core/rate_limit/test_limiter.py:639-640` inspect private buckets.
```python
assert limiter._buckets[0].count() == 2  # per-second bucket
assert limiter._buckets[1].count() == 2  # per-minute bucket
```

## Impact

- Tests can fail on refactors that preserve behavior but change internals; discourages encapsulation; reduces confidence that public contracts are correct.

## Root Cause Hypothesis

- No public hooks for introspection led to tests asserting internal state for convenience and to avoid slow time‑window waits.

## Recommended Fix

- Replace private state assertions with public‑behavior checks:
  - For `requests_per_minute=None`, assert construction succeeds and that `try_acquire` works within the per‑second limit without inspecting `_requests_per_minute`.
  - For registry configuration, assert expected limits via `try_acquire`/`acquire` behavior (e.g., consume N tokens then verify the next call fails) instead of `_requests_per_second`.
  - For `registry.close()`, assert new limiter instances are created after `close()` rather than reading `_limiters`.
  - For atomicity tests, spy on `pyrate_limiter.Limiter.try_acquire` (via `monkeypatch`) to ensure no limiter calls occur when the minute bucket is full, avoiding `_buckets` inspection.
---
# Test Defect Report

## Summary

- Some tests are assertion‑free and only rely on “no exception,” which weakens their verification of expected behavior.

## Severity

- Severity: trivial
- Priority: P3

## Category

- Weak Assertions

## Evidence

- `tests/core/rate_limit/test_limiter.py:379-387` has no assert statements.
```python
limiter = NoOpLimiter()

# Should not block or raise
limiter.acquire()
limiter.acquire(weight=100)
```
- `tests/core/rate_limit/test_limiter.py:406-411` has no assert statements.
```python
limiter = NoOpLimiter()
limiter.close()  # Should not raise
```

## Impact

- These tests can pass even if the methods return unexpected values or change behavior in subtle ways, giving false confidence.

## Root Cause Hypothesis

- Tests were written as smoke checks to ensure no exception, without explicit assertions.

## Recommended Fix

- Add explicit assertions on return values or post‑conditions to make expectations concrete, e.g., `assert limiter.acquire() is None` and `assert limiter.close() is None`, and optionally assert idempotency (calling twice does not raise).
