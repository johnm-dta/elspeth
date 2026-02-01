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

## Verification Status (2026-01-24)

**Status**: STILL VALID

**Verified by**: Automated verification agent

**Current code state**: The bug is confirmed to still exist in the current codebase. At lines 205-207 of `/home/john/elspeth-rapid/src/elspeth/plugins/sinks/database_sink.py`, the `write()` method uses:

```python
payload_json = json.dumps(rows, sort_keys=True, separators=(",", ":"))
payload_bytes = payload_json.encode("utf-8")
content_hash = hashlib.sha256(payload_bytes).hexdigest()
```

This is non-canonical JSON serialization. The codebase provides proper canonical JSON functions in `src/elspeth/core/canonical.py`:
- `canonical_json(obj)` - Two-phase normalization (pandas/numpy types) + RFC 8785 serialization
- `stable_hash(obj)` - Returns SHA-256 hash of canonical JSON

Recent git history shows the file has received security fixes (dd3bed7, e3c72a8) and refactoring (533ed86, 58685dd) but the canonical hashing issue has not been addressed.

**Architectural Impact**: The code comment at line 189 states "CRITICAL: Hashes the canonical JSON payload BEFORE insert" and line 57 states "Returns ArtifactDescriptor with SHA-256 hash of canonical JSON payload BEFORE insert" - but the implementation does not match this documented contract.

**Risks if unfixed**:
1. Hash divergence for semantically identical data (unicode escaping, field ordering)
2. NaN/Infinity values can pass through hashing (violates canonical JSON's strict rejection policy)
3. Pandas/numpy types will cause `TypeError` instead of being normalized (e.g., `pd.Timestamp`, `numpy.int64`)
4. Audit trail integrity compromised - hashes won't match `stable_hash()` computations elsewhere in the system

**Recommendation**: Keep open - this is a P1 audit integrity bug that needs fixing before production use.

## Fix Applied (2026-01-28)

**Status**: FIXED

**Fixed by**: Claude Code (fix/rc1-bug-burndown-session-6)

**Changes made**:

1. **`src/elspeth/plugins/sinks/database_sink.py`**:
   - Replaced `json.dumps` + `hashlib.sha256` with `stable_hash()` from `elspeth.core.canonical`
   - Replaced manual byte counting with `canonical_json().encode()` for payload_size
   - Removed unused `import hashlib` and `import json`
   - Added import for `canonical_json, stable_hash`

2. **`tests/plugins/sinks/test_database_sink.py`**:
   - Added new test class `TestDatabaseSinkCanonicalHashing` with 5 tests:
     - `test_content_hash_uses_canonical_json` - Unicode hashing (emoji, accents)
     - `test_content_hash_rejects_nan` - NaN rejection
     - `test_content_hash_rejects_infinity` - Infinity rejection
     - `test_content_hash_handles_numpy_types` - numpy.int64/float64 normalization
     - `test_payload_size_uses_canonical_bytes` - Correct byte counting
   - Updated existing tests to use `stable_hash()` instead of hardcoded `json.dumps`
   - Removed unused `import hashlib` and `import json`

**Verification**:
- All 27 database_sink tests pass
- All 79 sink tests pass
- mypy and ruff checks pass

**Breaking change**: Hash values for database artifacts will differ from previous versions when data contains unicode characters. This is correct behavior per RFC 8785.
