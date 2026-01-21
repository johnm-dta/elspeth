# Bug Report: DatabaseSink hashes non-canonical JSON payloads

## Summary

- DatabaseSink claims to hash canonical JSON payloads before insert, but it uses `json.dumps` instead of `canonical_json`, so hashes differ from the canonical standard and allow non-finite floats/un-normalized numpy/pandas types.

## Severity

- Severity: critical
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

1. Create a DatabaseSink with schema `{"fields": "dynamic"}` and a row containing non-ASCII text or a pandas/numpy type.
2. Compute `stable_hash(rows)` using `elspeth.core.canonical.stable_hash`.
3. Call `DatabaseSink.write(rows, ctx)` and compare `artifact.content_hash` to the canonical hash.
4. Observe mismatch (or `TypeError` for Decimal/pandas Timestamp), despite contract requiring canonical JSON hashing.

## Expected Behavior

- DatabaseSink hashes the RFC 8785 canonical JSON payload (via `canonical_json`/`stable_hash`), normalizing pandas/numpy types and rejecting NaN/Infinity.

## Actual Behavior

- DatabaseSink uses `json.dumps(..., sort_keys=True, separators=(",", ":"))`, which is not canonical JSON, does not normalize pandas/numpy types, and allows non-finite floats (producing non-standard JSON).

## Evidence

- `src/elspeth/plugins/sinks/database_sink.py` uses `json.dumps` for payload hashing.
- Contract requirement: `docs/contracts/plugin-protocol.md` states database content_hash is SHA-256 of canonical JSON payload before insert.
- Canonicalization rules: `src/elspeth/core/canonical.py`.

## Impact

- User-facing impact: Hashes can differ between runs for semantically identical data (ordering, unicode escaping), undermining reproducibility.
- Data integrity / security impact: Violates audit integrity rules; non-finite floats can be hashed even though canonical JSON forbids them.
- Performance or cost impact: Potential failures when rows contain pandas/numpy types that canonicalization would handle.

## Root Cause Hypothesis

- DatabaseSink bypasses canonicalization helpers and hashes raw `json.dumps` output.

## Proposed Fix

- Code changes (modules/files):
  - Use `canonical_json` (or `stable_hash`) in `src/elspeth/plugins/sinks/database_sink.py` to compute payload bytes/hash.
  - Ensure `payload_size` uses the canonical JSON byte length.
- Config or schema changes: None.
- Tests to add/update:
  - Add tests for canonical hashing with non-ASCII, numpy/pandas, and NaN/Infinity rejection.
- Risks or migration steps: Hash values for database artifacts will change; note in release notes.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (canonical JSON hashing for database artifacts), `src/elspeth/core/canonical.py`.
- Observed divergence: Uses `json.dumps` instead of canonical JSON.
- Reason (if known): Implementation shortcut in sink rewrite.
- Alignment plan or decision needed: Use canonical JSON helper consistently.

## Acceptance Criteria

- DatabaseSink content_hash matches `stable_hash(rows)` for supported inputs.
- Non-finite floats raise before hashing.
- Tests cover canonicalization edge cases.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_database_sink.py -k content_hash`
- New tests required: Yes (canonical hashing edge cases).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md` (canonical JSON)
