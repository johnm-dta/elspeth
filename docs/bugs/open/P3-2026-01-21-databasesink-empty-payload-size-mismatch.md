# Bug Report: DatabaseSink reports size_bytes=0 while hashing "[]" for empty writes

## Summary

- DatabaseSink computes content_hash from the JSON payload "[]" but returns `size_bytes=0` for empty writes, so size_bytes does not match the hashed payload length.

## Severity

- Severity: minor
- Priority: P3

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

1. Create a DatabaseSink with any schema (e.g., dynamic).
2. Call `sink.write([], ctx)`.
3. Observe `artifact.content_hash` equals SHA-256 of "[]" while `artifact.size_bytes` is `0`.

## Expected Behavior

- `size_bytes` should match the length of the payload that was hashed (2 bytes for "[]"), or the hash should be for empty content if size_bytes is 0.

## Actual Behavior

- `size_bytes` is set to 0 even though the hash is computed from "[]".

## Evidence

- `src/elspeth/plugins/sinks/database_sink.py` computes `payload_json = json.dumps(rows, ...)` then returns `payload_size=0` for empty rows.
- Contract: `docs/contracts/plugin-protocol.md` requires size_bytes for verification.

## Impact

- User-facing impact: Artifact metadata for empty writes is internally inconsistent.
- Data integrity / security impact: Verification tooling cannot reconcile `size_bytes` with the hashed content.
- Performance or cost impact: None.

## Root Cause Hypothesis

- Early return for empty rows overwrites `payload_size` with 0 while keeping the hash of "[]".

## Proposed Fix

- Code changes (modules/files):
  - Use the computed payload length even for empty rows, or compute the hash for empty content to match size_bytes=0.
- Config or schema changes: None.
- Tests to add/update:
  - Update empty-write tests to assert size_bytes matches payload length and hash.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (size_bytes required for verification).
- Observed divergence: size_bytes does not match hashed payload length.
- Reason (if known): Empty-write shortcut.
- Alignment plan or decision needed: Decide canonical size_bytes for empty payloads and make hash consistent.

## Acceptance Criteria

- For empty writes, size_bytes and content_hash are consistent (either both represent "[]" or both represent empty content).

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_database_sink.py::TestDatabaseSink::test_batch_write_empty_list -v`
- New tests required: Update existing empty-write test expectation.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
