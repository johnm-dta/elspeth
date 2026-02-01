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

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 6a

**Current Code Analysis:**

The bug is **confirmed present** in the current codebase. Examining `/home/john/elspeth-rapid/src/elspeth/plugins/sources/json_source.py` lines 184-188:

```python
# Extract from nested key if specified
if self._data_key:
    data = data[self._data_key]  # Line 185 - UNGUARDED ACCESS

if not isinstance(data, list):
    raise ValueError(f"Expected JSON array, got {type(data).__name__}")
```

**Problems identified:**

1. **Missing dict check**: If `data` is a list and `data_key` is set, line 185 will raise `TypeError: list indices must be integers or slices, not str` instead of quarantining
2. **Missing key check**: If `data` is a dict but `data_key` doesn't exist, line 185 will raise `KeyError` instead of quarantining

Both scenarios violate the Three-Tier Trust Model (CLAUDE.md) - external data structure errors should be quarantined, not crash the pipeline.

**Test coverage gap:**

The test suite has one test for `data_key` (line 118: `test_json_object_with_data_key`) but it only tests the happy path where the key exists and the root is a dict. No tests exist for:
- Missing `data_key` in JSON object
- `data_key` configured when root is a list
- `data_key` configured when root is neither dict nor list

**Git History:**

Related commits examined:
- `cec7dbb` (2026-01-23): Fixed `JSONDecodeError` handling in array files - different issue
- `ff2fcea` (earlier): Fixed JSONL parse errors - different issue

No commits address the `data_key` validation issue. The similar bug P1-2026-01-21-jsonsource-array-parse-errors-crash was closed as fixed (commit `cec7dbb`), but that only handled JSON parse errors, not structural mismatches with `data_key`.

**Root Cause Confirmed:**

YES. The unguarded dictionary access on line 185 will crash when:
1. User configures `data_key: "results"` but JSON file is `[{"id": 1}]` (list root) → `TypeError`
2. User configures `data_key: "results"` but JSON file is `{"items": [...]}` (key missing) → `KeyError`
3. External data changes structure (API returns list instead of object) → crash instead of graceful handling

This is a Tier 3 trust boundary violation - external data shape should not crash the pipeline.

**Recommendation:**

**Keep open** - bug is valid and needs fixing. The fix should:

1. Validate `data` is a dict before accessing `data_key`
2. Validate `data_key` exists in the dict
3. If either check fails, record validation error with `schema_mode="structure"` and quarantine/discard per config
4. Add tests for both failure scenarios (missing key, wrong root type)

Pattern should match existing `JSONDecodeError` handling (lines 160-181) which correctly treats parse errors as Tier 3 quarantine events.
