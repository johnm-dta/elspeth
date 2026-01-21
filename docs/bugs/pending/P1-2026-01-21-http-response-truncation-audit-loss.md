# Bug Report: AuditedHTTPClient truncates non-JSON responses in audit trail

## Summary

- For non-JSON responses, `AuditedHTTPClient` truncates the response body to 100,000 characters before recording. This violates the "full response recorded" audit requirement and makes replay/verify impossible for large or binary payloads.

## Severity

- Severity: critical
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
- Notable tool calls or steps: code inspection of audited HTTP client response handling

## Steps To Reproduce

1. Call an endpoint that returns a large non-JSON body (>100k bytes).
2. Observe the recorded response body in the audit trail is truncated to 100,000 characters.
3. Attempt replay/verify; the recorded response does not match the original payload.

## Expected Behavior

- The full response payload should be recorded (via payload store if necessary), preserving the complete body and hash.

## Actual Behavior

- Response bodies are truncated for non-JSON content, losing data.

## Evidence

- Truncation logic: `src/elspeth/plugins/clients/http.py:168-170`

## Impact

- User-facing impact: replay/verify cannot reproduce or validate large or binary responses.
- Data integrity / security impact: audit trail is incomplete; hashes refer to truncated payloads.
- Performance or cost impact: unclear; truncation may be hiding a need for proper payload storage.

## Root Cause Hypothesis

- Non-JSON responses are coerced to text and truncated instead of being stored as full payload bytes.

## Proposed Fix

- Code changes (modules/files):
  - Store full response bytes in the payload store and record metadata (size, content type) in `response_data`.
  - Consider base64 encoding for binary payloads if structured storage is required.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that records a large non-JSON response and asserts the full payload is recoverable.
- Risks or migration steps:
  - Ensure payload retention policies can handle large responses; document size implications.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (auditability standard: "External calls - Full request AND response recorded")
- Observed divergence: non-JSON responses are truncated.
- Reason (if known): size guardrail.
- Alignment plan or decision needed: decide on payload retention strategy for large responses.

## Acceptance Criteria

- Recorded HTTP responses preserve the full payload (or a recoverable full payload via payload store) regardless of size or content type.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k http_response`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
