# Bug Report: SanitizedWebhookUrl leaves secrets in URL fragments

## Summary

- SanitizedWebhookUrl.from_raw_url() sanitizes query parameters and basic auth but ignores URL fragments, allowing OAuth tokens and other fragment-based secrets to leak into the audit trail.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/contracts/url.py:153-179` - only inspects `parsed.query` and `parsed.username/password` for secrets
- `src/elspeth/contracts/url.py:235` - `parsed.fragment` is passed through unchanged in `urlunparse()`
- OAuth implicit flow commonly uses fragment tokens (e.g., `#access_token=xxx`)
- Module docstring at lines 3-7 guarantees "URLs cannot contain credentials when stored in the audit trail"

## Impact

- User-facing impact: URLs with fragment-based tokens (OAuth implicit flow) leak secrets
- Data integrity / security impact: Direct security violation - secrets in audit trail
- Performance or cost impact: None

## Root Cause Hypothesis

- URL fragment was not considered as a potential secret location during implementation.

## Proposed Fix

- Code changes:
  - In `SanitizedWebhookUrl.from_raw_url()`, check fragment for sensitive patterns (access_token, token, key, secret, etc.)
  - Either strip fragment entirely or sanitize sensitive parameters within it
- Tests to add/update:
  - Add test with `#access_token=secret` fragment asserting it's sanitized
  - Add test with benign fragment (e.g., `#section1`) asserting it's preserved

## Acceptance Criteria

- Fragment-based secrets (access_token, token, key, secret) are sanitized or removed
- Benign fragments (anchors like `#section1`) are preserved
- Module docstring guarantee is maintained
