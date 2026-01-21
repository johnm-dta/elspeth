# Bug Report: JSONSource crashes when data_key is missing or root is not an object

## Summary

- `_load_json_array` directly indexes `data[self._data_key]` without checking that the root is a dict or that the key exists.
- If the JSON root is a list or the key is missing, a `TypeError`/`KeyError` crashes the run instead of recording a validation error.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: JSON object without the configured `data_key`

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/sources`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Create a JSON file like `{"results": [{"id": 1}]}`.
2. Configure `JSONSource` with `data_key: "items"` (missing key) and `on_validation_failure: quarantine`.
3. Run the pipeline or call `JSONSource.load()`.

Alternate repro:
1. Create a JSON array file `[ {"id": 1} ]`.
2. Configure `JSONSource` with any non-empty `data_key`.
3. Run the pipeline.

## Expected Behavior

- Missing or invalid `data_key` is treated as a validation failure for external data.
- An audit record is written and the error is quarantined (or discarded if configured), not a crash.

## Actual Behavior

- `KeyError` (missing key) or `TypeError` (root is list) escapes `_load_json_array` and crashes the run.

## Evidence

- Direct indexing without validation: `src/elspeth/plugins/sources/json_source.py:153-156`

## Impact

- User-facing impact: external data shape changes (or config mistakes) crash ingestion rather than producing quarantine outputs.
- Data integrity / security impact: violates Tier 3 handling (external data should not crash the pipeline).
- Performance or cost impact: reruns and manual debugging required.

## Root Cause Hypothesis

- `data_key` handling assumes a well-formed JSON object and does not treat missing keys as an external data validation error.

## Proposed Fix

- Code changes (modules/files):
  - In `_load_json_array`, validate `data` is a dict and `self._data_key in data` before indexing.
  - If invalid, record a validation error (schema_mode="parse" or "structure") and quarantine/discard per config.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests for missing `data_key` and for `data_key` set on list roots.
- Risks or migration steps:
  - Decide whether to treat missing keys as fatal config errors vs external data errors; document expected behavior.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` Tier 3 external data handling
- Observed divergence: missing/invalid `data_key` crashes run instead of quarantine.
- Reason (if known): unguarded dictionary access.
- Alignment plan or decision needed: confirm policy for structural JSON mismatches.

## Acceptance Criteria

- Missing or invalid `data_key` produces a validation error record and quarantine/discard outcome, not a crash.

## Tests

- Suggested tests to run: `pytest tests/plugins/sources/test_json_source.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
