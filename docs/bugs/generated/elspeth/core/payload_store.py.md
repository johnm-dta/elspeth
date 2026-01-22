# Bug Report: Unvalidated content_hash allows path traversal outside base_path

## Summary

- FilesystemPayloadStore uses the raw `content_hash` to build filesystem paths, so a corrupted or attacker-controlled ref can escape `base_path` and read/delete arbitrary files, violating the audit trail "crash on anomaly" contract.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: 81a0925d7d6de0d0e16fdd2d535f63d096a7d052 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic x86_64 GNU/Linux
- Python version: Python 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit for payload_store.py
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only filesystem sandbox, network restricted, approvals never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Inspected payload_store implementation and usage references via rg/sed

## Steps To Reproduce

1. Create a temp base directory and an "outside" file in its parent.
2. Instantiate `FilesystemPayloadStore(base_path=<temp_dir>)`.
3. Call `store.exists(str(outside_path))` or `store.delete(str(outside_path))`.
4. Observe that the check/delete targets the outside file, not a path under `base_path`.

## Expected Behavior

- Invalid payload refs are rejected before filesystem access, and all operations are confined to `base_path`.

## Actual Behavior

- `content_hash` is used directly in path construction, allowing absolute or traversal paths to escape `base_path`.

## Evidence

- `src/elspeth/core/payload_store.py:103`
- `src/elspeth/core/payload_store.py:106`
- `src/elspeth/core/payload_store.py:141`
- `src/elspeth/core/payload_store.py:151`

## Impact

- User-facing impact: purge/explain can operate on unintended files if a corrupted ref reaches the payload store.
- Data integrity / security impact: path traversal enables arbitrary file deletion and potential data exposure.
- Performance or cost impact: Unknown.

## Root Cause Hypothesis

- The implementation assumes `content_hash` is always valid and trusted, but it never enforces the SHA-256 hex invariant or base_path containment.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/payload_store.py`: validate `content_hash` matches `^[0-9a-f]{64}$` before path construction; reject absolute paths and ensure resolved paths stay under `base_path` (raise `ValueError` or `IntegrityError` on violation).
- Config or schema changes: None.
- Tests to add/update:
  - Add tests in `tests/core/test_payload_store.py` asserting invalid hashes (absolute paths, "..", wrong length) raise and do not touch the filesystem.
- Risks or migration steps:
  - Existing rows with invalid refs will now fail fast (intended per audit integrity policy).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...):
  - `CLAUDE.md:40`
- Observed divergence:
  - Invalid internal refs do not cause an immediate crash and can trigger filesystem operations outside `base_path`.
- Reason (if known):
  - Validation was omitted in the payload store implementation.
- Alignment plan or decision needed:
  - Enforce strict hash validation to satisfy Tier 1 "crash on anomaly" requirements.

## Acceptance Criteria

- `retrieve/exists/delete` reject any `content_hash` not a 64-char lowercase hex string.
- Filesystem operations never escape `base_path` even with crafted input.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/test_payload_store.py`
- New tests required: Yes, invalid-hash/path traversal cases.

## Notes / Links

- Related issues/PRs: `docs/bugs/closed/P1-2026-01-19-payload-store-integrity-and-hash-validation-missing.md`
- Related design docs: `CLAUDE.md`
---
# Bug Report: store() skips integrity verification for existing blobs

## Summary

- `FilesystemPayloadStore.store()` treats any pre-existing blob as valid and writes directly to the final path; a corrupted or partially written file can be silently accepted and later fail retrieval, breaking auditability.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: 81a0925d7d6de0d0e16fdd2d535f63d096a7d052 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic x86_64 GNU/Linux
- Python version: Python 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit for payload_store.py
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only filesystem sandbox, network restricted, approvals never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Inspected payload_store implementation and usage references via rg/sed

## Steps To Reproduce

1. Create a temp payload store and store some content to get its hash.
2. Overwrite the on-disk file for that hash with different bytes.
3. Call `store(original_content)` again and note it returns the hash without error.
4. Call `retrieve(hash)` and observe `IntegrityError`.

## Expected Behavior

- If the hash path exists but content mismatches, `store()` should raise `IntegrityError` (or rewrite atomically if policy allows).
- Writes should be atomic to prevent partial blobs at the final path.

## Actual Behavior

- `store()` skips any verification when a file exists and writes directly to the final path, so corrupted/partial blobs can persist undetected until retrieval.

## Evidence

- `src/elspeth/core/payload_store.py:113`
- `src/elspeth/core/payload_store.py:116`
- `src/elspeth/core/payload_store.py:118`

## Impact

- User-facing impact: explain/replay/resume can fail with `IntegrityError` for payloads that were "successfully" stored earlier.
- Data integrity / security impact: audit trail contains refs that no longer resolve to their payloads.
- Performance or cost impact: Unknown.

## Root Cause Hypothesis

- `store()` assumes existence implies validity and does not validate or use atomic write semantics.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/payload_store.py`: if the target path exists, hash the existing bytes and compare to the computed hash; raise `IntegrityError` on mismatch.
  - Write to a temp file and `replace()` atomically after a successful write to prevent partial blobs.
- Config or schema changes: None.
- Tests to add/update:
  - Add a test in `tests/core/test_payload_store.py` ensuring `store()` raises on pre-existing mismatched content.
  - Add a test covering atomic write behavior (or simulate by pre-creating a truncated file and asserting store handles it explicitly).
- Risks or migration steps:
  - Existing corrupted blobs will now fail fast; operators may need a repair/purge workflow.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...):
  - `docs/design/architecture.md:565`
- Observed divergence:
  - Payload hashes are intended for integrity verification, but `store()` can accept a mismatched blob and only fail later at `retrieve()`.
- Reason (if known):
  - Idempotent write logic omits validation.
- Alignment plan or decision needed:
  - Enforce integrity checks on store and switch to atomic writes.

## Acceptance Criteria

- `store()` raises `IntegrityError` when an existing blob's bytes do not match the computed hash.
- Store writes are atomic and never leave partially written blobs at the final hash path.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/test_payload_store.py`
- New tests required: Yes, store-existing-mismatch and atomic write behavior.

## Notes / Links

- Related issues/PRs: `docs/bugs/closed/P1-2026-01-19-payload-store-integrity-and-hash-validation-missing.md`
- Related design docs: `docs/design/architecture.md`
