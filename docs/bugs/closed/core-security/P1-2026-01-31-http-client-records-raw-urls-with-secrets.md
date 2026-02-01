# Bug Report: AuditedHTTPClient records raw URLs containing secrets

## Summary

- `AuditedHTTPClient` stores the full URL in `request_data["url"]` without sanitization. URLs containing secrets in query parameters (e.g., `?token=sk-secret`) are recorded verbatim in the audit trail.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/clients/http.py:204-210`:
  - `full_url` stored directly in `request_data["url"]` with no sanitization
  - Headers ARE filtered (line 209 `_filter_request_headers`)
  - But URLs are not sanitized
- CLAUDE.md line 678: "Never store secrets - use HMAC fingerprints"

## Impact

- User-facing impact: Secrets visible in audit exports
- Data integrity / security impact: Direct security violation - API keys, tokens in audit trail
- Performance or cost impact: None

## Root Cause Hypothesis

- URL sanitization was not implemented when HTTP client auditing was added. Headers were sanitized but URLs were overlooked.

## Proposed Fix

- Code changes:
  - Create URL sanitization similar to `SanitizedWebhookUrl` for HTTP client URLs
  - Sanitize sensitive query parameters (token, key, secret, api_key, password, auth)
  - Store sanitized URL and fingerprint of sensitive params
- Tests to add/update:
  - Add test with URL containing `?api_key=secret`, verify secret not in audit

## Acceptance Criteria

- URLs with sensitive query parameters are sanitized before recording
- Sensitive parameters are replaced with fingerprints or redacted
- Non-sensitive URLs recorded unchanged
