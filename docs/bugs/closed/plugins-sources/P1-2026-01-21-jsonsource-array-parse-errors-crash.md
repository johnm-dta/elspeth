# Bug Report: JSONSource crashes on invalid JSON array files instead of quarantining

## Summary

- `_load_json_array` uses `json.load()` without handling `JSONDecodeError`.
- A malformed JSON file crashes the run, even though external data parsing errors should be quarantined at the source boundary.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: invalid JSON array file

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/sources`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Create a JSON file with invalid syntax, e.g. `[ {"id": 1}, {"id": 2` (missing closing `}]`).
2. Configure `JSONSource` with `format: "json"`, `schema: { fields: dynamic }`, and `on_validation_failure: quarantine`.
3. Run the pipeline or call `JSONSource.load()` with a `PluginContext`.

## Expected Behavior

- The malformed JSON is recorded as a validation error and quarantined (or discarded if configured).
- The run fails gracefully without an unhandled exception (even if no rows can be processed).

## Actual Behavior

- `json.JSONDecodeError` escapes `_load_json_array` and crashes the run.

## Evidence

- `_load_json_array` calls `json.load(f)` without a `try/except`: `src/elspeth/plugins/sources/json_source.py:149-156`

## Impact

- User-facing impact: a single malformed JSON file halts ingestion and yields no quarantine output for investigation.
- Data integrity / security impact: violates Tier 3 handling (external data should not crash the pipeline).
- Performance or cost impact: reruns and manual data cleaning required.

## Root Cause Hypothesis

- File-level JSON parse errors are not treated as quarantine events and are allowed to crash the run.

## Proposed Fix

- Code changes (modules/files):
  - Wrap `json.load` in `_load_json_array` with `try/except JSONDecodeError`.
  - Record a validation error (schema_mode="parse" or similar) and yield a quarantined row when `on_validation_failure != "discard"`.
  - Consider storing a truncated raw payload or a pointer to the source file for audit traceability.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test for invalid JSON array files to verify quarantine behavior (no crash).
- Risks or migration steps:
  - Ensure large JSON files do not bloat audit storage; consider truncation.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` Tier 3 external data handling
- Observed divergence: malformed JSON array file crashes the run instead of being quarantined.
- Reason (if known): parse errors not handled in `_load_json_array`.
- Alignment plan or decision needed: define audit representation for file-level parse failures.

## Acceptance Criteria

- Invalid JSON array files are recorded as validation errors and do not crash the pipeline.
- If `on_validation_failure != "discard"`, a quarantined record is produced for audit.

## Tests

- Suggested tests to run: `pytest tests/plugins/sources/test_json_source.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
