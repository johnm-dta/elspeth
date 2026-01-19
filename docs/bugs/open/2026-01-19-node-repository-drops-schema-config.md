# Bug Report: `NodeRepository.load()` drops `schema_mode` / `schema_fields` (WP-11.99 audit schema config lost on read)

## Summary

- Nodes store schema configuration for audit trail via `nodes.schema_mode` and `nodes.schema_fields_json` (WP-11.99).
- `NodeRepository.load()` constructs `Node(...)` without populating these fields, so consumers using the repository layer silently lose schema audit metadata.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `8ca061c9293db459c9a900f2f74b19b59a364a42`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive subsystem 4 (Landscape) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection

## Steps To Reproduce

1. Insert a `nodes` row with non-NULL `schema_mode` and `schema_fields_json`.
2. Load it via `NodeRepository.load(row)` (as used in repository patterns/tests).
3. Observe returned `Node.schema_mode` / `Node.schema_fields` are defaulted to `None`.

## Expected Behavior

- Repository layer returns complete Node objects, including schema configuration fields.

## Actual Behavior

- Schema config fields are silently dropped.

## Evidence

- Node schema includes audit schema configuration columns:
  - `src/elspeth/core/landscape/schema.py` (`nodes.schema_mode`, `nodes.schema_fields_json`)
- Node contract includes these fields:
  - `src/elspeth/contracts/audit.py:49-82`
- Repository load omits them:
  - `src/elspeth/core/landscape/repositories.py:65-88`
- Recorder has a separate `_row_to_node` path that *does* load them (inconsistent layering):
  - `src/elspeth/core/landscape/recorder.py:580-608`

## Impact

- User-facing impact: any UI/reporting that uses repository layer may omit schema audit details.
- Data integrity / security impact: low (data is stored), but audit explainability is reduced.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Repository module was not updated when WP-11.99 schema columns were added.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/repositories.py`:
    - Parse `schema_fields_json` (JSON) and populate `schema_fields`
    - Populate `schema_mode`
- Config or schema changes: none.
- Tests to add/update:
  - Add a repository test that includes schema_mode/schema_fields_json and asserts fields are preserved.
- Risks or migration steps:
  - None.

## Architectural Deviations

- Spec or doc reference: WP-11.99 (“Config-Driven Plugin Schemas”)
- Observed divergence: repository layer drops schema config on read.
- Reason (if known): missed update during schema feature addition.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- `NodeRepository.load()` preserves `schema_mode` and `schema_fields` for nodes when present.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_repositories.py -k NodeRepository`
- New tests required: yes (schema config preservation)

## Notes / Links

- Related issues/PRs: N/A
