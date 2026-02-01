# Bug Report: RateLimitRegistry not closed when resume fails

## Summary

- `_execute_resume_with_instances()` closes `RateLimitRegistry` only on success; if `orchestrator.resume()` raises, rate limiters are never closed.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/cli.py:1459-1485` - `rate_limit_registry.close()` after `orchestrator.resume()` with no try/finally

## Proposed Fix

- Wrap in try/finally to always close registry

## Acceptance Criteria

- RateLimitRegistry.close() called on both success and failure

## Resolution (2026-02-02)

**Status: FIXED**

Wrapped the resource creation and orchestrator.resume() call in try/finally at `src/elspeth/cli.py:1593-1634`:

```python
# Initialize to None so they're defined in finally block even if creation fails
rate_limit_registry = None
telemetry_manager = None

try:
    rate_limit_config = RuntimeRateLimitConfig.from_settings(config.rate_limit)
    rate_limit_registry = RateLimitRegistry(rate_limit_config)
    # ... orchestrator setup and resume ...
    return result
finally:
    # Clean up rate limit registry and telemetry (always, even on failure)
    if rate_limit_registry is not None:
        rate_limit_registry.close()
    if telemetry_manager is not None:
        telemetry_manager.close()
```

This mirrors the established pattern in `_execute_pipeline()`. All 123 resume-related tests pass.
