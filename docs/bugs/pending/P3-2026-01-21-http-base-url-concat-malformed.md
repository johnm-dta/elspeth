# Bug Report: AuditedHTTPClient base_url concatenation can create malformed URLs

## Summary

- `AuditedHTTPClient` concatenates `base_url` and `url` via string interpolation. Missing or extra slashes yield malformed URLs (e.g., `.../v1process` or double slashes), and absolute URLs get incorrectly prefixed.

## Severity

- Severity: minor
- Priority: P3

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
- Notable tool calls or steps: code inspection of URL handling

## Steps To Reproduce

1. Configure `base_url="https://api.example.com/v1"` and call `post("process", json={...})`.
2. Observe full URL becomes `https://api.example.com/v1process`.
3. Configure `base_url="https://api.example.com/"` and call `post("/v1/process", ...)`.
4. Observe double slash in URL (`https://api.example.com//v1/process`).

## Expected Behavior

- URL joining should be robust to leading/trailing slashes and absolute URLs.

## Actual Behavior

- URLs are concatenated naively, producing malformed or unintended endpoints.

## Evidence

- String concatenation of base URL and path: `src/elspeth/plugins/clients/http.py:134`

## Impact

- User-facing impact: requests can target wrong endpoints or fail with malformed URL errors.
- Data integrity / security impact: low.
- Performance or cost impact: low.

## Root Cause Hypothesis

- URL construction uses string concatenation instead of a proper URL join utility.

## Proposed Fix

- Code changes (modules/files):
  - Use `httpx.URL` or `urllib.parse.urljoin`, or configure `httpx.Client(base_url=...)` and pass relative paths.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests for base_url joining with and without leading/trailing slashes.
- Risks or migration steps:
  - Ensure existing callers that pass full URLs continue to work.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: N/A
- Reason (if known): simple concatenation.
- Alignment plan or decision needed: N/A

## Acceptance Criteria

- `AuditedHTTPClient` produces correct URLs for common `base_url` and `url` combinations.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k base_url`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
