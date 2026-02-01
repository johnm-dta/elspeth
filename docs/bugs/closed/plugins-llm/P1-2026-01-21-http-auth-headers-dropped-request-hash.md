# Bug Report: AuditedHTTPClient drops auth headers, causing request hash collisions

## Summary

- `AuditedHTTPClient` removes auth/key/token headers entirely from recorded request data, so calls that differ only by credentials hash to the same request. Replay/verify can return the wrong response and the audit trail cannot distinguish which credentials were used.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/clients` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of audited HTTP client

## Steps To Reproduce

1. Create an `AuditedHTTPClient` with the same `state_id` but different `Authorization` headers.
2. Call the same endpoint with identical JSON bodies.
3. Observe the recorded `request_data` omits the auth header, producing identical `request_hash` values.
4. In replay/verify mode, requests cannot be distinguished and the first response is reused.

## Expected Behavior

- Sensitive headers should be fingerprinted (HMAC) or redacted while still contributing to request identity, so calls with different credentials produce different hashes.

## Actual Behavior

- Sensitive headers are dropped entirely, causing request hash collisions and ambiguous audit records.

## Evidence

- Header filtering drops auth/key/token headers: `src/elspeth/plugins/clients/http.py:63-86`
- Filtered headers used in request data: `src/elspeth/plugins/clients/http.py:138-144`

## Impact

- User-facing impact: replay/verify can return a response tied to the wrong credentials.
- Data integrity / security impact: audit trail cannot prove which credentials were used for a call.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Sensitive headers are removed rather than fingerprinted, so they do not participate in hashing or audit lineage.

## Proposed Fix

- Code changes (modules/files):
  - Replace header removal with HMAC fingerprinting for sensitive values; store fingerprints in `request_data` so request hashes remain distinct without storing secrets.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that two calls with different auth headers produce different request hashes (but no raw secrets in recorded payloads).
- Risks or migration steps:
  - Ensure fingerprints use a stable, configured key so hashes remain consistent across runs.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` ("Secret Handling: Never store secrets - use HMAC fingerprints")
- Observed divergence: secrets are dropped instead of fingerprinted.
- Reason (if known): conservative redaction.
- Alignment plan or decision needed: adopt fingerprinting in request/response recording.

## Acceptance Criteria

- Requests with different auth headers produce distinct request hashes while never storing raw secret values.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k http_headers`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Current Implementation

Examined `/home/john/elspeth-rapid/src/elspeth/plugins/clients/http.py`:

1. **Header Filtering (lines 67-86)**: The `_filter_request_headers()` method completely removes sensitive headers from `request_data`:
   - Filters out headers in `_SENSITIVE_REQUEST_HEADERS` frozenset: `authorization`, `x-api-key`, `api-key`, `x-auth-token`, `proxy-authorization`
   - Also filters any header containing: `auth`, `key`, `secret`, or `token` (case-insensitive)

2. **Request Hash Calculation**:
   - Line 138-144: Filtered headers (without auth) are included in `request_data`
   - Line 2062 in `recorder.py`: `request_hash = stable_hash(request_data)`
   - The hash is computed from canonical JSON of the filtered data

3. **Impact Confirmed**: Two HTTP requests to the same URL with identical JSON bodies but different `Authorization` headers will:
   - Generate identical `request_hash` values
   - Be indistinguishable in the audit trail
   - Cause replay/verify to potentially return the wrong cached response

### Existing Infrastructure

**Good news**: The solution already exists but is not being used:

- `/home/john/elspeth-rapid/src/elspeth/core/security/fingerprint.py` implements `secret_fingerprint()` using HMAC-SHA256
- Function signature: `secret_fingerprint(secret: str, *, key: bytes | None = None) -> str`
- Returns 64-character hex digest that can be stored safely in audit trail
- Supports both environment variable (`ELSPETH_FINGERPRINT_KEY`) and Azure Key Vault for key management

**The fingerprinting module is NOT imported or used anywhere in `/src/elspeth/plugins/`**.

### Test Coverage

Reviewed `/home/john/elspeth-rapid/tests/plugins/clients/test_audited_http_client.py`:

- **Test exists** (line 126): `test_auth_headers_filtered_from_recorded_request()` validates that auth headers are filtered out
- **Missing test**: No test verifies that two requests with different auth headers produce different `request_hash` values
- The existing test confirms the bug behavior but doesn't test for the desired outcome

### Git History

Searched git history since 2026-01-21:
- No commits addressing auth header fingerprinting
- No commits modifying `http.py` since RC-1 release
- No references to this bug report ID in commit messages

### Root Cause Confirmed

The bug description is accurate:
1. Auth headers are completely dropped (not fingerprinted)
2. Request hashes are calculated from filtered data
3. This violates the architectural principle in `CLAUDE.md`: "Secret Handling: Never store secrets - use HMAC fingerprints"

### Recommendation

**Priority: P1 - Should be fixed before production use**

The fix requires:
1. Import `secret_fingerprint` from `elspeth.core.security` in `http.py`
2. Modify `_filter_request_headers()` to replace sensitive header values with fingerprints instead of removing them
3. Store as `"Authorization": f"<fingerprint:{fp}>"`  or similar redacted format that preserves uniqueness
4. Add test verifying different auth headers produce different request hashes
5. Verify fingerprints work correctly in replay mode (same auth = same fingerprint = cache hit)

---

## Resolution (2026-01-28)

**Status: FIXED**

### Changes Made

1. **Modified `src/elspeth/plugins/clients/http.py`:**
   - Added `os` import for environment variable access
   - Added new `_is_sensitive_header()` method to centralize header sensitivity detection
   - Rewrote `_filter_request_headers()` to fingerprint sensitive headers instead of removing them
   - When fingerprint key available: stores `<fingerprint:64hexchars>` format
   - When no key but dev mode (`ELSPETH_ALLOW_RAW_SECRETS=true`): removes headers (fail-safe)
   - When no key and not dev mode: removes headers with warning (production fail-safe)

2. **Added tests in `tests/plugins/clients/test_audited_http_client.py`:**
   - `test_auth_headers_fingerprinted_in_recorded_request`: Verifies fingerprint format and raw secrets not stored
   - `test_different_auth_headers_produce_different_request_hashes`: Verifies hash uniqueness with different credentials
   - `test_auth_headers_removed_when_no_fingerprint_key_dev_mode`: Verifies dev mode fallback behavior

### Verification

- All 25 HTTP client tests pass
- All 179 security and client tests pass
- All 3264 unit tests pass (12 expected skips for credential-gated tests)
- Type checking passes with mypy

### Commit

Branch: `fix/rc1-bug-burndown-session-6`
