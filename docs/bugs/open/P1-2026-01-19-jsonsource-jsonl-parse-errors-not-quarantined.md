# Bug Report: JSONSource crashes on invalid JSONL line (`JSONDecodeError`) instead of quarantining the row

## Summary

- `JSONSource` supports JSONL input (one JSON object per line) and is responsible for handling Tier-3 external data.
- In `_load_jsonl`, it calls `json.loads(line)` without catching `json.JSONDecodeError`.
- A single malformed JSON line crashes the entire run instead of being recorded and quarantined, contradicting the “quarantine bad external rows; continue processing” policy.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `8cfebea78be241825dd7487fed3773d89f2d7079`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any `.jsonl` with a malformed line

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 6 (plugins), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Create a `.jsonl` file with at least one malformed JSON line, e.g.:
   - Line 1: `{"id": 1}`
   - Line 2: `{bad json`
   - Line 3: `{"id": 2}`
2. Configure `JSONSource` with `format: "jsonl"` (or rely on `.jsonl` extension auto-detection) and set `on_validation_failure` to a quarantine sink name.
3. Run the pipeline.

## Expected Behavior

- The malformed line is treated as an invalid external row:
  - the error is recorded (audit trail)
  - the row is quarantined to `on_validation_failure` (unless `"discard"`)
  - processing continues for subsequent lines

## Actual Behavior

- `json.JSONDecodeError` escapes `_load_jsonl` and crashes the run.

## Evidence

- `_load_jsonl` performs `json.loads(line)` without a `try/except` around parse errors: `src/elspeth/plugins/sources/json_source.py:107-115`
- Quarantine machinery only handles `pydantic.ValidationError` during schema validation: `src/elspeth/plugins/sources/json_source.py:143-165`

## Impact

- User-facing impact: a single corrupt line in a large JSONL feed prevents any processing and produces no quarantine output for investigation.
- Data integrity / security impact: violates the stated Tier-3 handling contract (external data should not crash the pipeline).
- Performance or cost impact: increases reruns and manual data cleanup.

## Root Cause Hypothesis

- JSON parsing errors occur before schema validation and are not treated as “row-level invalid input”.

## Proposed Fix

- Code changes (modules/files):
  - In `_load_jsonl`, catch `json.JSONDecodeError` per line, record a validation error via `ctx.record_validation_error`, and emit `SourceRow.quarantined(...)` if `on_validation_failure != "discard"`.
  - Decide how to represent the failing row in `row_data` for audit:
    - e.g., `{"__raw_line__": line, "__line_number__": n}` or similar, since there is no parsed dict.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test for JSONL parse failure quarantining (malformed line does not crash; subsequent valid lines still load).
- Risks or migration steps:
  - Ensure raw line recording doesn’t leak secrets if JSONL can contain sensitive data; consider truncation limits in audit payloads.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (Tier 3 external data: quarantine invalid rows, continue)
- Observed divergence: invalid JSONL line crashes run instead of being quarantined.
- Reason (if known): parse errors not treated as row-level validation failures.
- Alignment plan or decision needed: define whether parse errors become `validation_errors` (recommended) and how raw lines are recorded.

## Acceptance Criteria

- Malformed JSONL lines are quarantined (or discarded if configured) and do not crash the pipeline.
- Subsequent valid lines are still processed.

## Tests

- Suggested tests to run: `pytest tests/plugins/sources/test_json_source.py`
- New tests required: yes

## Notes / Links

- Related design docs: `docs/design/requirements.md` (external data quarantine policy)
