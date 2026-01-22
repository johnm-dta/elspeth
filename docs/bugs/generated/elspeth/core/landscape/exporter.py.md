# Bug Report: Token export omits expand_group_id

## Summary

- `LandscapeExporter._iter_records()` emits `token` records without `expand_group_id`, so deaggregation lineage cannot be reconstructed from exported audit data.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux 6.8.0-90-generic (Ubuntu)
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/core/landscape/exporter.py
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled (never)
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `src/elspeth/core/landscape/exporter.py`, `src/elspeth/contracts/audit.py`, `src/elspeth/core/landscape/schema.py`

## Steps To Reproduce

1. Create a run that uses `LandscapeRecorder.expand_token(...)` so tokens get `expand_group_id`.
2. Call `LandscapeExporter.export_run(run_id)` and filter records where `record_type == "token"`.
3. Inspect token records and note that `expand_group_id` is missing.

## Expected Behavior

- Token records include `expand_group_id` (null when not applicable).

## Actual Behavior

- Token records omit `expand_group_id`.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/landscape/exporter.py:212`, `src/elspeth/core/landscape/exporter.py:219`, `src/elspeth/contracts/audit.py:111`, `src/elspeth/core/landscape/schema.py:107`
- Minimal repro input (attach or link): Use the Steps To Reproduce; any run with `expand_token` shows the missing field in export output.

## Impact

- User-facing impact: auditors cannot reconstruct deaggregation groupings from exported audit data alone.
- Data integrity / security impact: audit completeness gap (lineage metadata missing).
- Performance or cost impact: None.

## Root Cause Hypothesis

- The token export mapping was not updated when `expand_group_id` was added to tokens.

## Proposed Fix

- Code changes (modules/files): add `expand_group_id` to token records in `src/elspeth/core/landscape/exporter.py`.
- Config or schema changes: None.
- Tests to add/update: add an exporter test that creates expanded tokens and asserts `expand_group_id` is present.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/contracts/audit.py:111`
- Observed divergence: export omits a contract field required for deaggregation lineage.
- Reason (if known): mapping omission.
- Alignment plan or decision needed: include `expand_group_id` in exported token records.

## Acceptance Criteria

- Exported `token` records always include `expand_group_id` with correct value or null.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_exporter.py`
- New tests required: yes, cover expanded-token export.

## Notes / Links

- Related issues/PRs: `docs/bugs/open/P2-2026-01-19-exporter-missing-expand-group-id.md`
- Related design docs: Unknown
---
# Bug Report: Export omits run/node configuration and determinism metadata

## Summary

- Run and node records omit configuration and determinism fields (`settings_json`, `config_json`, `determinism`, schema config), so exported audit data is not self-contained for configuration traceability.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux 6.8.0-90-generic (Ubuntu)
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/core/landscape/exporter.py
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled (never)
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `src/elspeth/core/landscape/exporter.py`, `src/elspeth/contracts/audit.py`, `src/elspeth/core/landscape/schema.py`, `CLAUDE.md`

## Steps To Reproduce

1. Run any pipeline and complete a run.
2. Export the run via `LandscapeExporter.export_run(run_id)`.
3. Inspect `record_type == "run"` and `record_type == "node"` records and note missing config/determinism fields.

## Expected Behavior

- Exported run and node records include configuration JSON and determinism/schema metadata required to trace decisions.

## Actual Behavior

- Export includes hashes only and omits `settings_json`, `config_json`, `determinism`, and schema config fields.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/landscape/exporter.py:162`, `src/elspeth/core/landscape/exporter.py:173`, `src/elspeth/contracts/audit.py:38`, `src/elspeth/contracts/audit.py:62`, `src/elspeth/contracts/audit.py:64`, `src/elspeth/core/landscape/schema.py:34`, `src/elspeth/core/landscape/schema.py:57`, `src/elspeth/core/landscape/schema.py:59`, `CLAUDE.md:17`
- Minimal repro input (attach or link): Any completed run export; inspect run/node records for missing fields.

## Impact

- User-facing impact: exported audit trail cannot stand alone for compliance review without separate access to original configuration artifacts.
- Data integrity / security impact: reduced traceability to configuration (audit completeness gap).
- Performance or cost impact: None (unless config inclusion is later added).

## Root Cause Hypothesis

- Exporter mapping was implemented with a minimal field set and not updated to include config/determinism/schema metadata.

## Proposed Fix

- Code changes (modules/files): include `settings_json` in run records and `config_json`, `determinism`, `schema_mode`, `schema_fields` (and optionally export status metadata) in node/run records in `src/elspeth/core/landscape/exporter.py`.
- Config or schema changes: None.
- Tests to add/update: add exporter tests asserting presence of config/determinism/schema fields.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:17`
- Observed divergence: export does not carry configuration needed for traceability.
- Reason (if known): exporter field set not kept in sync with audit schema/contracts.
- Alignment plan or decision needed: decide if export must be self-contained; if yes, include config/determinism/schema fields by default or via a flag.

## Acceptance Criteria

- Run records include `settings_json` and node records include `config_json` and `determinism` (plus schema config fields), and tests validate these fields.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_exporter.py`
- New tests required: yes, validate config/determinism/schema fields in export output.

## Notes / Links

- Related issues/PRs: `docs/bugs/pending/P2-2026-01-19-exporter-missing-config-in-export.md`
- Related design docs: Unknown
---
# Bug Report: Exporter uses N+1 query pattern across row/token/state hierarchy

## Summary

- `LandscapeExporter._iter_records()` performs nested per-entity queries, leading to a large number of DB round-trips and slow exports for large runs.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux 6.8.0-90-generic (Ubuntu)
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/core/landscape/exporter.py
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled (never)
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `src/elspeth/core/landscape/exporter.py` and query methods in `src/elspeth/core/landscape/recorder.py`

## Steps To Reproduce

1. Execute a run with thousands of rows, multiple tokens per row, and multiple node states per token.
2. Export the run via `LandscapeExporter.export_run(run_id)`.
3. Observe export time and DB query count balloon due to nested per-entity queries.

## Expected Behavior

- Export should use a bounded number of queries per record type (batch loads), keeping export time roughly linear with record count.

## Actual Behavior

- Export issues many small queries in nested loops, leading to poor scalability.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/landscape/exporter.py:199`, `src/elspeth/core/landscape/exporter.py:211`, `src/elspeth/core/landscape/exporter.py:234`, `src/elspeth/core/landscape/exporter.py:286`, `src/elspeth/core/landscape/exporter.py:300`, `src/elspeth/core/landscape/recorder.py:1738`, `src/elspeth/core/landscape/recorder.py:1758`, `src/elspeth/core/landscape/recorder.py:1909`, `src/elspeth/core/landscape/recorder.py:1939`
- Minimal repro input (attach or link): Large synthetic run (thousands of rows/tokens/states) exported via `export_run`.

## Impact

- User-facing impact: exports become slow or unusable for large runs.
- Data integrity / security impact: low.
- Performance or cost impact: high DB overhead and potential lock contention.

## Root Cause Hypothesis

- Exporter composes per-entity recorder methods inside nested loops, resulting in N+1 query patterns.

## Proposed Fix

- Code changes (modules/files): batch-load rows/tokens/states/events/calls in `src/elspeth/core/landscape/exporter.py` using fewer queries and in-memory grouping; preserve deterministic ordering for signing.
- Config or schema changes: None.
- Tests to add/update: add a performance regression test or query-count benchmark (optional).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: export does not scale linearly with record volume due to query strategy.
- Reason (if known): exporter uses naive nested fetch pattern.
- Alignment plan or decision needed: define acceptable export performance targets and implement batching.

## Acceptance Criteria

- Export completes with a bounded number of queries per record type and maintains deterministic ordering for signatures.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_exporter.py`
- New tests required: optional (benchmark or query-count guard).

## Notes / Links

- Related issues/PRs: `docs/bugs/pending/P2-2026-01-19-exporter-n-plus-one-queries.md`
- Related design docs: Unknown
