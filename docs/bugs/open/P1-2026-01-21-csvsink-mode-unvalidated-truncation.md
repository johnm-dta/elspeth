# Bug Report: CSVSink accepts invalid mode values and silently truncates

## Summary

- `CSVSinkConfig.mode` is an unconstrained string, so typos (e.g., "apend") are accepted and treated as write mode, silently truncating existing files.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6f088f467276582fa8016f91b4d3bb26c7 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive into src/elspeth/plugins/sinks for bugs.
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): Codex CLI, workspace-write sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Manual code inspection only

## Steps To Reproduce

1. Create a CSV file with existing data at `output.csv`.
2. Configure `CSVSink` with `mode: "apend"` (typo).
3. Call `sink.write([...], ctx)`.
4. Observe the file is opened in write mode and truncated.

## Expected Behavior

- Invalid mode values should be rejected during configuration validation.

## Actual Behavior

- Any non-"append" value falls back to write behavior, risking silent data loss.

## Evidence

- `src/elspeth/plugins/sinks/csv_sink.py` checks `if self._mode == "append"` and otherwise uses write mode.
- `CSVSinkConfig.mode` is declared as `str` with no validation.

## Impact

- User-facing impact: A simple typo can wipe existing output files.
- Data integrity / security impact: Data loss in audit artifacts.
- Performance or cost impact: None.

## Root Cause Hypothesis

- Config model does not constrain `mode` to allowed values.

## Proposed Fix

- Code changes (modules/files):
  - Change `CSVSinkConfig.mode` to `Literal["write", "append"]` and validate.
  - Consider raising explicit error for unknown values in `_open_file`.
- Config or schema changes: None.
- Tests to add/update:
  - Add config validation test for invalid mode values.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: Invalid config values are accepted silently.
- Reason (if known): Missing validation.
- Alignment plan or decision needed: Enforce strict config validation.

## Acceptance Criteria

- Invalid `mode` values raise `PluginConfigError` during initialization.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_csv_sink.py -k mode`
- New tests required: Add invalid-mode validation test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
