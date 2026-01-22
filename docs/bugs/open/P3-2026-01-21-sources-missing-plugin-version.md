# Bug Report: CSVSource and JSONSource report plugin_version as 0.0.0

## Summary

- `CSVSource` and `JSONSource` do not set `plugin_version`, so they inherit the base default `"0.0.0"`.
- The orchestrator records plugin metadata from the instance; audit records show incorrect versions for these core sources.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any run using CSVSource or JSONSource

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/sources`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Run a pipeline that uses `CSVSource` or `JSONSource`.
2. Inspect the `nodes.plugin_version` value in Landscape or export.

## Expected Behavior

- Source nodes record a real semantic version (e.g., `"1.0.0"`), consistent with other built-in plugins.

## Actual Behavior

- Source nodes record `"0.0.0"` because the class attribute is never set.

## Evidence

- `CSVSource` has no `plugin_version` attribute: `src/elspeth/plugins/sources/csv_source.py`
- `JSONSource` has no `plugin_version` attribute: `src/elspeth/plugins/sources/json_source.py`
- Base default is `"0.0.0"`: `src/elspeth/plugins/base.py:292-300`
- Orchestrator records instance metadata: `src/elspeth/engine/orchestrator.py:580-599`

## Impact

- User-facing impact: audit metadata for core sources is misleading or uninformative.
- Data integrity / security impact: weaker reproducibility guarantees (cannot tie outputs to source plugin version accurately).
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `CSVSource` and `JSONSource` omitted `plugin_version` while other built-ins set it explicitly.

## Proposed Fix

- Code changes (modules/files):
  - Add `plugin_version = "1.0.0"` (or actual version) to `CSVSource` and `JSONSource`.
- Config or schema changes: none.
- Tests to add/update:
  - Add a metadata test ensuring built-in sources expose non-default `plugin_version`.
- Risks or migration steps:
  - Update any tests that assume `0.0.0` (unlikely).

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (plugin_version required for auditability)
- Observed divergence: core sources report a placeholder version.
- Reason (if known): oversight during source implementation.
- Alignment plan or decision needed: define versioning policy for core plugins.

## Acceptance Criteria

- `CSVSource` and `JSONSource` expose explicit, non-default `plugin_version` values.
- Audit records show these versions for source nodes.

## Tests

- Suggested tests to run: `pytest tests/plugins/sources/test_csv_source.py`, `pytest tests/plugins/sources/test_json_source.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
