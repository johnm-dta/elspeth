# Bug Report: ArtifactDescriptor leaks secrets via raw URLs

## Summary

- `ArtifactDescriptor.for_database` (and `for_webhook`) embeds raw URLs into `path_or_uri`, so when sinks pass credentialed URLs (e.g., database DSNs or tokenized webhook URLs), secrets are persisted into the audit trail and surfaced via exports/TUI, violating the secret-handling requirement.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `/home/john/elspeth-rapid/src/elspeth/contracts/results.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox; approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `src/elspeth/contracts/results.py`, `src/elspeth/plugins/sinks/database_sink.py`, `src/elspeth/core/landscape/recorder.py`, `CLAUDE.md`

## Steps To Reproduce

1. Configure `DatabaseSink` with a URL containing credentials (e.g., `postgresql://user:secret@host/db`).
2. Run any pipeline that writes rows to the database sink.
3. Inspect the artifacts table or exported lineage; observe `path_or_uri` includes the full URL with credentials.

## Expected Behavior

- Artifact records should never store secrets; URLs in `path_or_uri` should be sanitized or replaced with fingerprints.

## Actual Behavior

- `path_or_uri` contains the raw URL (including credentials or tokens), which is stored in the audit database.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/contracts/results.py:224` constructs `path_or_uri=f"db://{table}@{url}"`; `src/elspeth/contracts/results.py:242` constructs `path_or_uri=f"webhook://{url}"`; `src/elspeth/plugins/sinks/database_sink.py:205` passes `self._url` directly; `src/elspeth/core/landscape/recorder.py:1627` persists `path_or_uri` into the artifacts table.
- Minimal repro input (attach or link): Database URL with credentials (e.g., `postgresql://user:secret@host/db`) in sink config.

## Impact

- User-facing impact: Secrets can appear in lineage exports, TUI displays, or audit DB queries.
- Data integrity / security impact: High-risk secret leakage into the audit trail (non-redactable by policy).
- Performance or cost impact: Negligible.

## Root Cause Hypothesis

- `ArtifactDescriptor.for_database` and `for_webhook` directly embed raw URLs into `path_or_uri` without sanitization or fingerprinting, and sinks pass runtime URLs that may include secrets.

## Proposed Fix

- Code changes (modules/files): In `src/elspeth/contracts/results.py`, sanitize URLs before embedding in `path_or_uri` (strip credentials/query tokens) and optionally store a fingerprint in `metadata` for traceability; reuse existing sanitization/fingerprint helpers or add a dedicated helper in this module.
- Config or schema changes: None required if sanitization happens within `ArtifactDescriptor` factories.
- Tests to add/update: Update `tests/contracts/test_results.py` to assert sanitized URLs; add tests ensuring credentials/tokens are not present and fingerprints are captured when available.
- Risks or migration steps: Artifact `path_or_uri` format will change; update any tests or tooling that assert the old raw-URL format.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:358` (Secret Handling: “Never store secrets - use HMAC fingerprints”).
- Observed divergence: Raw credentialed URLs are stored in artifacts via `path_or_uri`.
- Reason (if known): Unknown.
- Alignment plan or decision needed: Sanitize and/or fingerprint URLs in `ArtifactDescriptor` factories to keep audit trail secret-safe.

## Acceptance Criteria

- Artifact records no longer contain credentials or tokens in `path_or_uri` for database/webhook artifacts.
- Tests demonstrate sanitization/fingerprinting behavior for URLs with secrets.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/contracts/test_results.py`
- New tests required: Yes, add coverage for sanitized/fingerprinted URLs.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:358`
