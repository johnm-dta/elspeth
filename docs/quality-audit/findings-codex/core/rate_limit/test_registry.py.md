# Test Defect Report

## Summary

- Tests claiming to validate service-specific/default rate limit configuration only assert type/name, not the configured rate values.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/rate_limit/test_registry.py:122` through `tests/core/rate_limit/test_registry.py:160` only check the limiter type and name, even though the test names and comments say they validate configuration:
  ```python
  def test_uses_service_specific_config(self) -> None:
      ...
      limiter = registry.get_limiter("openai")
      ...
      assert isinstance(limiter, RateLimiter)
      assert limiter.name == "openai"
  ```
  ```python
  def test_uses_default_config_for_unconfigured_service(self) -> None:
      ...
      limiter = registry.get_limiter("unknown_api")
      assert isinstance(limiter, RateLimiter)
      assert limiter.name == "unknown_api"
  ```
  No assertions verify `requests_per_second` or `requests_per_minute`.
- `src/elspeth/core/rate_limit/registry.py:91` through `src/elspeth/core/rate_limit/registry.py:99` shows the registry wiring config values into `RateLimiter`, which the tests never validate.
- `src/elspeth/core/rate_limit/limiter.py:143` through `src/elspeth/core/rate_limit/limiter.py:146` stores `requests_per_second` and `requests_per_minute` on the limiter, making them directly assertable.

## Impact

- A regression where `RateLimitRegistry` ignores per-service overrides or default limits would still pass these tests.
- Misconfigured limits could cause external API overuse (rate-limit violations) or overly aggressive throttling, and the tests would not catch it.
- The test names suggest coverage that does not actually exist, creating false confidence.

## Root Cause Hypothesis

- Tests avoid asserting private attributes and settle for type/name checks, leaving configuration wiring unverified despite being the core behavior under test.

## Recommended Fix

- In `tests/core/rate_limit/test_registry.py:122` and `tests/core/rate_limit/test_registry.py:144`, assert the configured rates on the returned limiter.
- Example pattern:
  ```python
  assert limiter._requests_per_second == 5
  assert limiter._requests_per_minute == 100
  ```
  and for defaults:
  ```python
  assert limiter._requests_per_second == 15
  assert limiter._requests_per_minute is None
  ```
- Priority justification: these are critical-path configuration checks for external rate limiting; weak assertions here can mask serious operational regressions.
