# Bug Report: ArtifactDescriptor accepts duck-typed sanitized URLs via hasattr

## Summary

- `ArtifactDescriptor.for_database()` and `.for_webhook()` use `hasattr()` checks instead of strict type enforcement, allowing any object with `sanitized_url` attribute to pass - potentially including unsanitized URLs with secrets.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/contracts/results.py:331-332` and `362-363` - uses `hasattr(url, "sanitized_url")` / `hasattr(url, "fingerprint")` for duck-typed acceptance.
- CLAUDE.md prohibits defensive patterns like `hasattr` for system-owned code
- Creates security risk: unsanitized URLs could reach audit trail

## Impact

- User-facing impact: Confusing type errors if wrong object passed
- Data integrity/security: Could allow credentials in audit trail

## Proposed Fix

- Replace `hasattr` checks with strict type enforcement for `SanitizedDatabaseUrl`/`SanitizedWebhookUrl`

## Acceptance Criteria

- Only verified Sanitized* types accepted
- TypeError raised for duck-typed objects

## Verification (2026-02-01)

**Status: STILL VALID**

- `ArtifactDescriptor.for_database()` and `.for_webhook()` still rely on `hasattr` checks instead of strict type enforcement. (`src/elspeth/contracts/results.py:331-333`, `src/elspeth/contracts/results.py:362-365`)
