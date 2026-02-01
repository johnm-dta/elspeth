# Bug Report: Run allows missing payload_store, violating source payload audit requirement

## Summary

- `run()` accepts `payload_store=None` and does not enforce payload persistence, allowing runs that omit raw source payload storage and violate the auditability standard.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-30
- Related run/issue ID: N/A

## Environment

- Commit/branch: 2678b83d1bef5b1ab2049b9babe625f4fb0b2799 (fix/P2-aggregation-metadata-hardcoded)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Unknown

## Steps To Reproduce

1. Instantiate `Orchestrator` and call `run()` with `payload_store=None`.
2. Process at least one row.
3. Inspect rows in the audit DB: `source_data_ref` is null and raw payload is not stored.

## Expected Behavior

- `run()` should reject execution without a payload store (or otherwise guarantee source payload persistence) to meet audit requirements.

## Actual Behavior

- `run()` proceeds with `payload_store=None`, and `LandscapeRecorder.create_row()` skips payload persistence when no payload store is configured.

## Evidence

- **Original issue (now fixed):** `payload_store` was optional and not validated in `run()`.
- **Current code (fix in place):** `run()` requires `payload_store` and raises if missing. (`src/elspeth/engine/orchestrator.py:524-553`)

## Impact

- User-facing impact: Audit exports and `explain()` cannot reconstruct original source inputs.
- Data integrity / security impact: Violates non-negotiable audit requirement ("Source entry - Raw data stored before any processing").
- Performance or cost impact: None directly, but increases audit risk.

## Root Cause Hypothesis

- Orchestrator treats payload storage as optional despite auditability standard requiring raw source payload persistence.

## Proposed Fix

- Code changes (modules/files):
  - In `src/elspeth/engine/orchestrator.py`, enforce `payload_store is not None` in `run()` with a clear error.
  - Or add a config flag like `require_payload_store=True` (default) that can be disabled for testing only.
- Config or schema changes: None.
- Tests to add/update:
  - Add a test that `run()` raises when `payload_store` is missing.
  - Add a test that with a payload store configured, `source_data_ref` is populated.
- Risks or migration steps:
  - Some test or dev flows may need to instantiate a payload store (expected, per audit requirements).

## Acceptance Criteria

- `run()` raises a clear error if `payload_store` is not provided (unless explicitly disabled for testing).
- Runs with a payload store always populate `source_data_ref` for source rows.

## Verification (2026-02-01)

**Status: FIXED**

- `Orchestrator.run()` now requires `payload_store` and raises `ValueError` when missing. (`src/elspeth/engine/orchestrator.py:524-553`)

## Closure Report (2026-02-01)

**Status:** CLOSED (FIXED)

### Closure Notes

- Payload store requirement is enforced in `Orchestrator.run()`, aligning with auditability requirements.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/unit/ -k "payload_store_required"`
- New tests required: yes, as described above.
