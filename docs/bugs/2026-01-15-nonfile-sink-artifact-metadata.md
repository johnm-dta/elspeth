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
  - `src/elspeth/engine/executors.py`: branch on `artifact_info["kind"]` (or presence of keys) and map to `path_or_uri`, `content_hash`, `size_bytes` appropriately.
  - `src/elspeth/engine/adapters.py`: for non-file sinks, provide `path`, `content_hash`, `size_bytes` with sensible defaults (e.g., `path=url`, `content_hash=""`, `size_bytes=0`) to satisfy Landscape schema.
- Config or schema changes: none (unless schema expands for non-file artifacts).
- Tests to add/update:
  - Add a test that uses the database sink and asserts run completes and artifact is registered.
- Risks or migration steps:
  - Decide how to represent non-file artifact identity in `path_or_uri` (URL vs table notation).

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
