# Bug Report: Key Vault empty secret accepted as HMAC key

## Summary

- `get_fingerprint_key()` accepts an empty Key Vault secret value and returns `b""`, which effectively removes the secret from HMAC and undermines the “no guessing oracle” guarantee for fingerprints.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of `src/elspeth/core/security/fingerprint.py`.
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): Read-only sandbox, approvals disabled.
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: `sed`, `rg`, `nl`, `date`.

## Steps To Reproduce

1. Create an Azure Key Vault secret named `elspeth-fingerprint-key` with an empty value.
2. Set `ELSPETH_KEYVAULT_URL` to that vault and ensure `ELSPETH_FINGERPRINT_KEY` is unset.
3. Call `get_fingerprint_key()` or `secret_fingerprint()` without providing a key.

## Expected Behavior

- An empty Key Vault secret is treated as invalid, and `get_fingerprint_key()` raises a `ValueError` just as it does when the key is missing.

## Actual Behavior

- `get_fingerprint_key()` returns `b""` for the empty Key Vault secret and `secret_fingerprint()` uses an empty HMAC key.

## Evidence

- `src/elspeth/core/security/fingerprint.py:88` only checks for `None`, not empty string.
- `src/elspeth/core/security/fingerprint.py:90` encodes and returns the empty string as the key.
- `docs/design/architecture.md:747` through `docs/design/architecture.md:753` require HMAC with a managed secret key to avoid guessing oracles.

## Impact

- User-facing impact: Fingerprinting appears to succeed, masking the misconfiguration.
- Data integrity / security impact: Empty HMAC key makes fingerprints effectively guessable (no secret key), defeating the intended protection against offline guessing.
- Performance or cost impact: None known.

## Root Cause Hypothesis

- Key Vault path only rejects `None` values; empty strings are treated as valid keys, unlike the env var path which rejects empty values by truthiness.

## Proposed Fix

- Code changes (modules/files): In `src/elspeth/core/security/fingerprint.py`, treat `secret.value == ""` as invalid (raise `ValueError`), mirroring the env var behavior.
- Config or schema changes: None.
- Tests to add/update: Add a test to `tests/core/security/test_fingerprint.py` asserting Key Vault empty string raises `ValueError`.
- Risks or migration steps: None; only affects misconfigured deployments.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:747` through `docs/design/architecture.md:753`.
- Observed divergence: Empty Key Vault secrets bypass validation and effectively remove the secret from HMAC, recreating the guessing-oracle risk HMAC is meant to prevent.
- Reason (if known): Missing empty-string validation in Key Vault retrieval.
- Alignment plan or decision needed: Align Key Vault validation with env var handling by rejecting empty values.

## Acceptance Criteria

- `get_fingerprint_key()` raises `ValueError` when Key Vault secret value is empty.
- New test covers the empty-string Key Vault case and passes.

## Tests

- Suggested tests to run: `pytest tests/core/security/test_fingerprint.py -v`
- New tests required: Yes, add coverage for empty Key Vault secret value.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `docs/design/architecture.md`
