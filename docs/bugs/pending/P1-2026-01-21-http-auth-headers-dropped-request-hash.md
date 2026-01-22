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
