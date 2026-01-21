# Bug Report: Artifact contract/recorder ignore artifacts.idempotency_key column

## Summary

- Landscape schema includes an `artifacts.idempotency_key` column “for retry deduplication”, but:
  - `contracts.audit.Artifact` has no `idempotency_key` field
  - `LandscapeRecorder.register_artifact()` does not accept or persist `idempotency_key`
  - `LandscapeRecorder.get_artifacts()` does not read it back
- Sink protocol docs describe idempotency keys, so the audit trail is missing a planned deduplication primitive.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Steps To Reproduce

N/A (feature is structurally unreachable today because it is not plumbed).

## Expected Behavior

- When a sink write is idempotent/deduplicated, the artifact record includes a stable `idempotency_key` so retries can be detected/audited.
- Artifact contracts reflect the schema columns that affect audit/retry semantics.

## Actual Behavior

- `idempotency_key` is defined in the database schema but cannot be recorded or queried through the public recorder/contract API.

## Evidence

- DB schema column exists:
  - `src/elspeth/core/landscape/schema.py:182-200` (`Column("idempotency_key", String(256))`)
- Artifact contract lacks the field:
  - `src/elspeth/contracts/audit.py:225-238`
- Recorder does not persist/read it:
  - `src/elspeth/core/landscape/recorder.py:1584-1641` (`register_artifact()` insert omits idempotency_key)
  - `src/elspeth/core/landscape/recorder.py:1643-1680` (`get_artifacts()` ignores idempotency_key)
- Sink protocol docs mention idempotency keys:
  - `src/elspeth/plugins/protocols.py:382-385`

## Impact

- User-facing impact: retries/dedup semantics are harder to reason about and audit.
- Data integrity / security impact: medium: lack of key means dedup decisions can’t be recorded/verified.
- Performance or cost impact: potential duplicate sink writes if retries occur and sinks attempt dedup without audit support.

## Root Cause Hypothesis

- Schema added an idempotency column, but contracts/recorder plumbing was never completed.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/contracts/audit.py`: add `idempotency_key: str | None` to `Artifact`.
  - `src/elspeth/core/landscape/recorder.py`: accept `idempotency_key` in `register_artifact()` and write to DB; read it in `get_artifacts()`.
  - `src/elspeth/engine/executors.py`: compute/pass an idempotency key for sink writes (likely `{run_id}:{row_id}:{sink_name}` or a hash thereof), aligned with sink protocol.
- Tests to add/update:
  - Add tests asserting `idempotency_key` is persisted and returned for artifacts when provided.

## Architectural Deviations

- Spec or doc reference: sink idempotency notes in `src/elspeth/plugins/protocols.py`
- Observed divergence: schema/documentation indicate idempotency, but audit contract does not represent it
- Alignment plan or decision needed: choose canonical idempotency key scheme and whether it is mandatory when `sink.idempotent=True`

## Acceptance Criteria

- `register_artifact(..., idempotency_key=...)` stores the key.
- `get_artifacts()` returns `Artifact` objects with `idempotency_key` populated.
- At least one sink execution path produces a non-null idempotency_key in artifacts records.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/landscape -k artifact`
- New tests required: yes

## Resolution

**Status:** FIXED

**Date:** 2026-01-21

**Changes made:**

1. `src/elspeth/contracts/audit.py`: Added `idempotency_key: str | None = None` field to `Artifact` dataclass
2. `src/elspeth/core/landscape/recorder.py`:
   - `register_artifact()`: Added `idempotency_key` parameter and writes to DB
   - `get_artifacts()`: Now reads and returns `idempotency_key` from DB

**Tests added:**

- `test_register_artifact_with_idempotency_key`: Verifies key is persisted and returned
- `test_register_artifact_without_idempotency_key_returns_none`: Verifies optional field behavior

**Note:** Acceptance criteria #3 (sink execution path producing idempotency_key) is deferred - sinks can now record idempotency keys via the plumbed API, but automatic key generation is a follow-on enhancement.
