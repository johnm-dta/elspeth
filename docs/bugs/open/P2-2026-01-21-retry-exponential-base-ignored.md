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
