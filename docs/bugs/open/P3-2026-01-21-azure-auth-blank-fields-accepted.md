# Bug Report: AzureAuthConfig accepts blank credential/account fields

## Summary

- `AzureAuthConfig.validate_auth_method()` only checks for `None`, not empty/whitespace strings, for `account_url`, `tenant_id`, `client_id`, and `client_secret`.
- Misconfigured auth (e.g., `account_url: ""` or `tenant_id: "   "`) passes validation but fails later during client creation, delaying failure and obscuring root cause.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/azure` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/plugins/azure/auth.py`

## Steps To Reproduce

1. Configure Azure auth with `use_managed_identity: true` and `account_url: ""` (or a whitespace-only string).
2. Initialize `AzureAuthConfig` via Azure source/sink config.
3. Run a pipeline using AzureBlobSource/Sink.

## Expected Behavior

- Configuration validation fails fast with a clear error about blank required fields.

## Actual Behavior

- Validation passes (fields are not `None`), and runtime client creation later fails with less actionable Azure errors.

## Evidence

- Validation checks only for `is not None` (no `strip()` or length checks):
  - `src/elspeth/plugins/azure/auth.py:85`
  - `src/elspeth/plugins/azure/auth.py:86`
  - `src/elspeth/plugins/azure/auth.py:87`
  - `src/elspeth/plugins/azure/auth.py:88`
  - `src/elspeth/plugins/azure/auth.py:125`

## Impact

- User-facing impact: confusing runtime failures instead of immediate config errors.
- Data integrity / security impact: none direct, but misconfigurations can be harder to diagnose.
- Performance or cost impact: failed runs and retries.

## Root Cause Hypothesis

- Auth validation treats empty strings as configured values and does not enforce non-empty content.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/azure/auth.py`: require non-empty strings for `account_url`, `tenant_id`, `client_id`, `client_secret` using `.strip()` checks or Pydantic `min_length=1`.
- Config or schema changes: none.
- Tests to add/update:
  - Add validation tests that reject whitespace-only fields for all auth methods.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): N/A
- Observed divergence: config validation is weaker than the usual fail-fast pattern for system-owned config.
- Reason (if known): validation focuses on presence, not content.
- Alignment plan or decision needed: enforce non-empty auth fields.

## Acceptance Criteria

- Blank or whitespace-only auth fields are rejected at config validation time.

## Tests

- Suggested tests to run:
  - `pytest tests/plugins/azure/test_auth.py`
- New tests required: yes (blank field validation)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
