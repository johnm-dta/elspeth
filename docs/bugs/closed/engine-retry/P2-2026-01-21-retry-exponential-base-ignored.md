# Bug Report: RetrySettings.exponential_base is ignored by RetryManager

## Summary

- `RetrySettings.exponential_base` is defined in config but never wired into `RetryConfig` or `wait_exponential_jitter`, so changing it has no effect on backoff behavior.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6f088f467276582fa8016f91b4d3bb26c7 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive into src/elspeth/engine/retry.py for bugs.
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): Codex CLI, workspace-write sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Manual code inspection only

## Steps To Reproduce

1. Configure `RetrySettings(exponential_base=10.0)`.
2. Run a retrying operation and observe the backoff growth.
3. Observe behavior matches tenacity default `exp_base=2` (no change).

## Expected Behavior

- `exponential_base` should control the exponent base in `wait_exponential_jitter`.

## Actual Behavior

- `RetryConfig.from_settings()` ignores `exponential_base`, and `RetryManager` always uses tenacity's default exponent base (2).

## Evidence

- `src/elspeth/core/config.py` defines `RetrySettings.exponential_base`.
- `src/elspeth/engine/retry.py` `RetryConfig` has no `exponential_base` and `execute_with_retry` calls `wait_exponential_jitter` without `exp_base`.
- Tenacity `wait_exponential_jitter` accepts `exp_base`.

## Impact

- User-facing impact: Retry backoff tuning via config is ineffective.
- Data integrity / security impact: None.
- Performance or cost impact: Mis-tuned retries can amplify load or slow recovery.

## Root Cause Hypothesis

- Config value exists but is not mapped into retry runtime configuration.

## Proposed Fix

- Code changes (modules/files):
  - Add `exponential_base` to `RetryConfig` and map from `RetrySettings`.
  - Pass `exp_base` into `wait_exponential_jitter`.
- Config or schema changes: None.
- Tests to add/update:
  - Add test asserting `RetryConfig` preserves `exponential_base` and `RetryManager` uses it.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/core/config.py` (RetrySettings includes `exponential_base`).
- Observed divergence: Value never affects backoff.
- Reason (if known): Incomplete mapping from settings to runtime.
- Alignment plan or decision needed: Wire config to tenacity.

## Acceptance Criteria

- Changing `exponential_base` changes backoff progression as expected.

## Tests

- Suggested tests to run: `pytest tests/engine/test_retry.py -k from_settings`
- New tests required: Yes.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 5

**Current Code Analysis:**

The bug is still present in the current codebase. Detailed examination reveals:

1. **RetrySettings (config.py:557-566)** defines `exponential_base` field:
   ```python
   exponential_base: float = Field(default=2.0, gt=1.0, description="Exponential backoff base")
   ```

2. **RetryConfig dataclass (retry.py:47-101)** has NO `exponential_base` field:
   - Only has: `max_attempts`, `base_delay`, `max_delay`, `jitter`
   - Missing: `exponential_base`

3. **RetryConfig.from_settings() (retry.py:87-101)** ignores `exponential_base`:
   ```python
   return cls(
       max_attempts=settings.max_attempts,
       base_delay=settings.initial_delay_seconds,
       max_delay=settings.max_delay_seconds,
       jitter=1.0,  # Fixed jitter, not exposed in settings
   )
   # exponential_base is NEVER read from settings
   ```

4. **RetryManager.execute_with_retry() (retry.py:155-159)** never passes `exp_base`:
   ```python
   wait=wait_exponential_jitter(
       initial=self._config.base_delay,
       max=self._config.max_delay,
       jitter=self._config.jitter,
       # exp_base is MISSING - uses tenacity default (2.0)
   )
   ```

5. **Verified tenacity API accepts exp_base parameter:**
   ```
   wait_exponential_jitter(initial: float = 1, max: float = ..., exp_base: float = 2, jitter: float = 1)
   ```

**Git History:**

Reviewed all retry-related commits:
- Commit `443114a` added `RetryConfig.from_settings()` but did NOT map `exponential_base`
- Commit `f2f3e2b` initially implemented `RetryManager` without `exponential_base`
- No subsequent commits addressed this missing wiring

**Root Cause Confirmed:**

The config field exists (`RetrySettings.exponential_base`), and the tenacity API supports it (`wait_exponential_jitter(..., exp_base=...)`), but the mapping code in `RetryConfig.from_settings()` and the execution code in `execute_with_retry()` never wire them together.

This is a classic incomplete-integration bug: the field was added to config schema but never plumbed through the runtime implementation.

**Recommendation:**

**Keep open** - This is a valid P2 bug that should be fixed. The fix is straightforward:

1. Add `exponential_base: float = 2.0` field to `RetryConfig` dataclass
2. Map it in `from_settings()`: `exponential_base=settings.exponential_base`
3. Pass it in `execute_with_retry()`: `wait_exponential_jitter(..., exp_base=self._config.exponential_base)`
4. Add test coverage to verify the mapping works end-to-end

Impact: Without this fix, users configuring custom exponential backoff bases will have their settings silently ignored, potentially causing performance issues or increased load on external services during retry storms.
