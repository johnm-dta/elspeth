# Bug Report: Landscape defines `calls` table but provides no API to record external calls (audit gap)

## Summary

- The Landscape schema includes a `calls` table (`calls_table`) intended to record external calls (LLM/HTTP/SQL/filesystem) with request/response hashes and optional payload refs.
- `LandscapeRecorder` exposes `get_calls(...)` and both `LandscapeExporter` and `explain()` attempt to include call records, but there is **no recorder method** to insert call records into the database.
- As a result, the audit trail cannot satisfy the project’s explicit requirement that external calls record full request/response at the boundary.

## Severity

- Severity: critical
- Priority: P0

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
- Notable tool calls or steps: static code inspection across `src/elspeth/core/landscape/*`

## Steps To Reproduce

1. Search for any call-recording API on `LandscapeRecorder` (e.g., `record_call`, `begin_call`, `complete_call`) and observe none exist.
2. Search for any insert path into `calls_table` and observe no writes occur from application code.
3. Run any pipeline that makes external calls (Phase 6) and inspect the audit DB: the `calls` table will remain empty because nothing inserts records.

## Expected Behavior

- The system provides a first-class recorder API to persist external call records, including:
  - `call_type`, `status`
  - request/response hashes
  - optional request/response payload refs (payload store)
  - error payload and latency
- Executors/plugins performing external calls (LLM, HTTP, etc.) use this API so the audit DB is complete.

## Actual Behavior

- `calls_table` exists and is queried, but there is no supported way to record calls through the Landscape subsystem.

## Evidence

- Schema defines call records as a first-class audit table:
  - `src/elspeth/core/landscape/schema.py:162-182` (`calls_table = Table("calls", ...)`)
- Recorder only supports querying calls, not recording them:
  - `src/elspeth/core/landscape/recorder.py:1922-1957` (only `select(calls_table)`)
- Exporter expects call records to exist:
  - `src/elspeth/core/landscape/exporter.py:297-318` (emits `"record_type": "call"` for each `get_calls(...)`)
- Explain/lineage expects call records to exist:
  - `src/elspeth/core/landscape/lineage.py:108-117`
- Explicit audit requirement (external boundary capture):
  - `CLAUDE.md` (“External calls - Full request AND response recorded”)

## Impact

- User-facing impact: explain/export outputs omit external call details, reducing the value of audit output and undermining “why did this decision happen?” investigations.
- Data integrity / security impact: high. Missing boundary records are a direct auditability failure.
- Performance or cost impact: N/A (feature gap).

## Root Cause Hypothesis

- Schema and query/explain/export layers were implemented before the recording API and the engine integration for Phase 6 external calls.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/recorder.py`:
    - Add `record_call(...)` (or begin/complete call) that inserts into `calls_table` with required fields and strict enum validation.
  - Integrate call recording at external system boundaries:
    - LLM provider adapter(s), HTTP client wrapper(s), SQL execution wrapper(s), filesystem reads/writes as appropriate.
- Config or schema changes:
  - Confirm how payload refs are stored (payload store) and ensure `request_ref`/`response_ref` are populated when configured.
- Tests to add/update:
  - Add an integration test that records a call via the new API and asserts:
    - row is written to `calls` table
    - `get_calls()` returns it with enum types (see related bug)
    - `LandscapeExporter` includes it in export output
- Risks or migration steps:
  - None for schema (table already exists). Must ensure per-call indexing (`state_id`, `call_index`) matches executor semantics.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (non-negotiable storage points)
- Observed divergence: external boundary records are not recorded in Landscape.
- Reason (if known): Phase 6 call recording not wired through LandscapeRecorder yet.
- Alignment plan or decision needed: ensure Phase 6 work includes the Landscape recording API and integration points.

## Acceptance Criteria

- External calls executed during a run are persisted to `calls` with request/response hashes and relevant metadata.
- `explain()` and `LandscapeExporter` include call records for affected node states.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/`
- New tests required: yes (call recording end-to-end)

## Notes / Links

- Related issues/PRs: `docs/bugs/open/2026-01-19-retention-purge-ignores-call-and-reason-payload-refs.md` (retention assumes call payload refs exist)
- Related design docs: `docs/design/architecture.md`
