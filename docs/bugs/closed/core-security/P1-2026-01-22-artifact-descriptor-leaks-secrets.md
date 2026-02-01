# Bug Report: ArtifactDescriptor leaks secrets via raw URLs

## Summary

`ArtifactDescriptor.for_database` and `for_webhook` embed raw URLs into `path_or_uri`, so when sinks pass credentialed URLs (e.g., database DSNs or tokenized webhook URLs), secrets are persisted into the audit trail and surfaced via exports/TUI, violating the secret-handling requirement.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: Codex (static analysis agent)
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: main (d8df733)
- OS: Linux
- Python version: 3.12+
- Config profile / env vars: Pipeline with DatabaseSink or webhook sink using credentialed URLs
- Data set or fixture: Any data flowing to database/webhook sink

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `src/elspeth/contracts/results.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox; approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed results.py, database_sink.py, recorder.py, CLAUDE.md

## Steps To Reproduce

1. Configure `DatabaseSink` with a URL containing credentials (e.g., `postgresql://user:secret@host/db`)
2. Run any pipeline that writes rows to the database sink
3. Inspect the artifacts table or exported lineage
4. Observe `path_or_uri` includes the full URL with credentials

## Expected Behavior

- Artifact records should never store secrets
- URLs in `path_or_uri` should be sanitized or replaced with fingerprints

## Actual Behavior

- `path_or_uri` contains the raw URL (including credentials or tokens), which is stored in the audit database

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots):
  - `src/elspeth/contracts/results.py:224` constructs `path_or_uri=f"db://{table}@{url}"`
  - `src/elspeth/contracts/results.py:242` constructs `path_or_uri=f"webhook://{url}"`
  - `src/elspeth/plugins/sinks/database_sink.py:205` passes `self._url` directly
  - `src/elspeth/core/landscape/recorder.py:1627` persists `path_or_uri` into the artifacts table
- Minimal repro input (attach or link): Database URL with credentials (e.g., `postgresql://user:secret@host/db`) in sink config

## Impact

- User-facing impact: Secrets can appear in lineage exports, TUI displays, or audit DB queries
- Data integrity / security impact: High-risk secret leakage into the audit trail (non-redactable by policy)
- Performance or cost impact: Negligible

## Root Cause Hypothesis

`ArtifactDescriptor.for_database` and `for_webhook` directly embed raw URLs into `path_or_uri` without sanitization or fingerprinting, and sinks pass runtime URLs that may include secrets.

## Proposed Fix

- Code changes (modules/files): In `src/elspeth/contracts/results.py`, sanitize URLs before embedding in `path_or_uri` (strip credentials/query tokens) and optionally store a fingerprint in `metadata` for traceability. Reuse existing sanitization/fingerprint helpers or add a dedicated helper in this module.
- Config or schema changes: None required if sanitization happens within `ArtifactDescriptor` factories
- Tests to add/update: Update `tests/contracts/test_results.py` to assert sanitized URLs; add tests ensuring credentials/tokens are not present and fingerprints are captured when available
- Risks or migration steps: Artifact `path_or_uri` format will change; update any tests or tooling that assert the old raw-URL format

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:358` - "Never store secrets - use HMAC fingerprints"
- Observed divergence: Raw credentialed URLs are stored in artifacts via `path_or_uri`
- Reason (if known): Unknown
- Alignment plan or decision needed: Sanitize and/or fingerprint URLs in `ArtifactDescriptor` factories to keep audit trail secret-safe

## Acceptance Criteria

- Artifact records no longer contain credentials or tokens in `path_or_uri` for database/webhook artifacts
- Tests demonstrate sanitization/fingerprinting behavior for URLs with secrets

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/contracts/test_results.py`
- New tests required: Yes, add coverage for sanitized/fingerprinted URLs

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:358`

## Verification Status

- [x] Bug confirmed via reproduction
- [x] Root cause verified
- [x] Fix implemented
- [x] Tests added
- [x] Fix verified

## Resolution

**Fixed in commit:** (this commit)
**Fixed by:** Claude Opus 4.5
**Date:** 2026-01-23

### Solution Implemented

Used **type-level enforcement (Option C)** per architecture critic recommendation:

1. Created `SanitizedDatabaseUrl` and `SanitizedWebhookUrl` frozen dataclass types in `src/elspeth/core/security/url.py`
2. Modified `ArtifactDescriptor.for_database()` and `for_webhook()` to require sanitized URL types
3. Updated `DatabaseSink` to sanitize URL at construction time
4. Added 44 new tests for URL sanitization

### Files Changed

| File | Change |
|------|--------|
| `src/elspeth/core/security/url.py` | NEW - Sanitized URL types |
| `src/elspeth/core/security/__init__.py` | Export new types |
| `src/elspeth/contracts/results.py` | Factory methods require sanitized types |
| `src/elspeth/plugins/sinks/database_sink.py` | Use sanitized URL for artifacts |
| `tests/core/security/test_url.py` | NEW - 44 tests |
| `tests/contracts/test_results.py` | Updated for new types |

### Key Design Decisions

- **Type enforcement** makes it impossible to accidentally pass raw URLs (mypy catches errors)
- **Reuses existing** `_sanitize_dsn()` infrastructure
- **Fingerprints only secrets**, not full URLs (for traceability across different endpoints)
- **Handles Basic Auth** in webhook URLs (`user:pass@host`)
- **Expanded sensitive params** list (OAuth, signed URLs, API keys)
