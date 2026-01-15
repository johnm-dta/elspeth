# Bug Report: Database/webhook sinks crash artifact registration

## Summary
- SinkAdapter returns non-file artifact metadata for database/webhook sinks, but SinkExecutor always expects file-style keys (`path`, `content_hash`, `size_bytes`), so runs with non-file sinks raise `KeyError` during artifact registration.

## Severity
- Severity: major
- Priority: P1

## Reporter
- Name or handle: Codex
- Date: 2026-01-15
- Related run/issue ID: N/A

## Environment
- Commit/branch: 5c27593 (local)
- OS: Linux (dev env)
- Python version: 3.11+ (per project)
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if applicable)
- Goal or task prompt: N/A
- Model/version: N/A
- Tooling and permissions (sandbox/approvals): N/A
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: N/A

## Steps To Reproduce
1. Configure a sink with `plugin: database` (or a webhook-style sink once available).
2. Run a pipeline via `elspeth` CLI using that sink.
3. Observe a `KeyError` when `SinkExecutor` registers the artifact.

## Expected Behavior
- Non-file sinks should register artifacts without crashing, using schema-appropriate fields.

## Actual Behavior
- `SinkExecutor` indexes `artifact_info["path"]`, `artifact_info["content_hash"]`, and `artifact_info["size_bytes"]`, but `SinkAdapter` returns `{"kind": "database", "url": ..., "table": ...}` for database sinks.

## Evidence
- Non-file artifact descriptors returned by adapter: `src/elspeth/engine/adapters.py`.
- Artifact registration requires file-style fields: `src/elspeth/engine/executors.py:780` (path/content_hash/size_bytes), `src/elspeth/core/landscape/recorder.py:1240`.

## Impact
- User-facing impact: Database/webhook sinks cannot be used without runtime failure.
- Data integrity / security impact: Audit trail cannot record sink artifacts for non-file destinations.
- Performance or cost impact: N/A (fails before completion).

## Root Cause Hypothesis
- SinkAdapter supports multiple artifact kinds, but SinkExecutor assumes file-only artifact metadata.

## Proposed Fix
- Code changes (modules/files):
  - Define an explicit Artifact contract at the sink boundary (best‑practice, audit‑safe):
    - Introduce an `ArtifactResult`/`ArtifactDescriptor` dataclass used by all sinks and adapters with required fields: `artifact_type`, `path_or_uri`, `content_hash`, `size_bytes`, and optional `metadata` for kind‑specific details.
    - `src/elspeth/engine/executors.py`: compute `content_hash` and `size_bytes` from the canonicalized sink input payload when the sink does not provide a stronger, authoritative digest. This avoids “fake defaults” and keeps all artifacts auditable, even for non‑file sinks.
    - `src/elspeth/engine/adapters.py`: return a fully populated `ArtifactDescriptor` (not a raw dict) and never emit partial keys; adapters should map sink identity into a canonical URI scheme:
      - files: `file:///path/to/output.csv`
      - databases: `db://<dsn>/<table>`
      - webhooks: `webhook://<url>`
    - `src/elspeth/core/landscape/recorder.py`: enforce the artifact contract (fail fast if missing `path_or_uri` or `content_hash`) instead of accepting placeholders.
  - Optional (recommended for long‑term auditability): add an `artifact_meta_json` column to store sink‑specific identity details without overloading `path_or_uri`.
- Config or schema changes: only if adding `artifact_meta_json` (otherwise keep schema and enforce required fields).
- Tests to add/update:
  - End‑to‑end test for database and webhook sinks ensuring:
    - `path_or_uri` uses the canonical scheme,
    - `content_hash` is non‑empty and stable for the same payload,
    - audit records are present without runtime exceptions.
  - Unit tests for ArtifactDescriptor serialization and canonical hashing.
- Risks or migration steps:
  - Requires clarifying whether `content_hash` represents *payload* (recommended) vs *storage bytes* (file‑only). If file sinks need file‑byte hash, document and/or add `artifact_meta_json` to store both without ambiguity.

## Architectural Deviations
- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/subsystems/00-overview.md` (artifacts table schema requires `path_or_uri`, `content_hash`, `size_bytes`).
- Observed divergence: database sink adapters do not provide these fields, causing runtime errors.
- Reason (if known): artifact kind handling not implemented in SinkExecutor.
- Alignment plan or decision needed: define standard mapping for non-file sink artifacts.

## Acceptance Criteria
- Pipelines using database sinks complete without KeyError and record an artifact row.

## Tests
- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py -k artifact`
- New tests required: yes (database sink artifact registration).

## Notes / Links
- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`
