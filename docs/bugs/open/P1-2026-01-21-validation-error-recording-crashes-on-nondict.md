# Bug Report: Validation error recording crashes on non-dict or non-finite rows

## Summary

- Quarantined source rows are explicitly allowed to be non-dict (primitives or malformed structures).
- `PluginContext.record_validation_error()` computes `stable_hash(row)` and `LandscapeRecorder.record_validation_error()` canonicalizes `row_data` as a dict.
- Non-dict rows or rows containing NaN/Infinity trigger `TypeError`/`ValueError`, crashing the run instead of quarantining and recording the error.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into contents of `src/elspeth/plugins` and create bug tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/plugins/context.py` and `src/elspeth/core/landscape/recorder.py`

## Steps To Reproduce

1. Instantiate a `PluginContext` with a `LandscapeRecorder` (or even with `landscape=None`).
2. Call `ctx.record_validation_error(row=42, error="bad", schema_mode="dynamic", destination="discard")`.
3. Observe a crash when `stable_hash()` or `canonical_json()` is invoked.

Alternative repro (non-finite values):
1. Call `ctx.record_validation_error(row={"value": float("nan")}, ...)` from any source.
2. Observe `ValueError: Cannot canonicalize non-finite float`.

## Expected Behavior

- Validation errors should be recorded and quarantined even when the row is non-dict or contains non-finite values.
- The pipeline should continue processing other rows per the Tier-3 trust model.

## Actual Behavior

- Recording a validation error raises `TypeError`/`ValueError` during hashing/canonicalization, aborting the run.

## Evidence

- `SourceRow` explicitly allows `row: Any` for quarantined data: `src/elspeth/contracts/results.py`.
- `PluginContext.record_validation_error()` hashes the raw row before any guard: `src/elspeth/plugins/context.py`.
- `LandscapeRecorder.record_validation_error()` requires `row_data: dict[str, Any]` and calls `canonical_json(row_data)`: `src/elspeth/core/landscape/recorder.py`.

## Impact

- User-facing impact: malformed external data crashes the entire run instead of being quarantined.
- Data integrity / security impact: audit trail missing for invalid inputs; violates Tier-3 trust boundary guarantees.
- Performance or cost impact: runs abort early; reprocessing required.

## Root Cause Hypothesis

- Validation error recording assumes rows are dicts and canonically serializable, but quarantined rows are explicitly allowed to be non-dict and may include non-finite values.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/context.py`: guard `stable_hash(row)` with a serialization-safe fallback for row_id generation (e.g., hash `repr(row)` if canonicalization fails).
  - `src/elspeth/core/landscape/recorder.py`: accept `row_data: Any` for validation errors and handle serialization failures by storing a safe raw representation (or payload-store ref) plus an explicit serialization-error marker.
- Tests to add/update:
  - Add a unit test that calls `record_validation_error()` with a primitive row and asserts no crash + audit record created.
  - Add a unit test with `float("nan")` in row data that results in a quarantined audit record, not a crash.
- Risks or migration steps:
  - Decide and document how non-canonical row data is represented in the audit trail (repr, payload ref, or explicit error wrapper).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md` Three-Tier Trust Model (Tier 3 requires quarantine, not crash).
- Observed divergence: invalid external data can crash during validation error recording.
- Reason (if known): validation error recorder assumes dict/canonical data even for quarantined rows.
- Alignment plan or decision needed: define canonical handling for non-dict/non-finite invalid rows in audit storage.

## Acceptance Criteria

- `record_validation_error()` handles non-dict and non-finite rows without raising.
- Audit records are written for quarantined rows even when data is malformed.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/landscape/test_validation_errors.py`
- New tests required: yes (non-dict + non-finite validation error recording)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
