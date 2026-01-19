# Bug Report: Payload store lacks integrity verification and content hash validation (path traversal risk)

## Summary

- `FilesystemPayloadStore` claims “integrity verification on retrieval” but `retrieve()` does not verify the on-disk bytes match the requested hash.
- `retrieve()/exists()/delete()` accept arbitrary `content_hash` strings and use them directly in filesystem paths, enabling path traversal if a corrupted/tampered ref (or any untrusted input) reaches the store.
- `store()` is non-atomic and does not validate existing content; a partial/corrupted file can be silently returned forever.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 3 (core infrastructure) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/core/payload_store.py`

## Steps To Reproduce

### A) Integrity verification is missing

1. Create a `FilesystemPayloadStore` with a temp `base_path`.
2. Call `store(b"hello")` and capture the returned hash.
3. Corrupt the stored file on disk (overwrite bytes).
4. Call `retrieve(<hash>)`.

### B) Path traversal via unvalidated `content_hash`

1. Create a `FilesystemPayloadStore(base_path=<some dir>)`.
2. Ensure a target file exists outside `base_path` (e.g., in `base_path.parent`).
3. Call `retrieve()` (or `delete()`) with a crafted `content_hash` containing path separators / `..` segments (example: `"../../target_file"`).
4. Observe the resolved path can escape `base_path` and read/delete unintended files (depends on FS layout).

## Expected Behavior

- `retrieve(hash)` verifies the returned bytes hash to `hash` and raises a hard error on mismatch.
- `retrieve()/exists()/delete()` reject invalid hashes (only allow 64-char lowercase hex).
- `store()` writes atomically and validates on-disk content when a hash path already exists.

## Actual Behavior

- `retrieve()` reads bytes and returns them without verifying content integrity.
- `content_hash` is used directly to construct paths; invalid hashes are not rejected.
- `store()` can leave partially written blobs and will treat any pre-existing file at the hash path as valid.

## Evidence

- Integrity verification is claimed but not implemented:
  - `src/elspeth/core/payload_store.py:5-8` (docstring claims “Integrity verification on retrieval”)
  - `src/elspeth/core/payload_store.py:107-112` (`retrieve()` returns `path.read_bytes()` with no hash check)
- Path is derived from unvalidated input:
  - `src/elspeth/core/payload_store.py:90-93` (`_path_for_hash()` uses `content_hash` directly)
- Non-atomic write:
  - `src/elspeth/core/payload_store.py:100-104` (`write_bytes()` to final path with no atomic rename)

## Impact

- User-facing impact: corrupted payloads can be returned as if valid; explain/export/replay can surface incorrect data without a clear failure mode.
- Data integrity / security impact: high. Undetected payload corruption undermines the audit trail. Path traversal risk can enable unintended reads/deletes if refs are ever attacker-controlled or the audit DB is tampered.
- Performance or cost impact: corrupted blobs may cause repeated retries/reprocessing; retention/purge safety is reduced.

## Root Cause Hypothesis

- The implementation computes hashes on write but never validates them on read.
- The interface assumes `content_hash` is always well-formed and trusted, but the code does not enforce the invariant.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/payload_store.py`:
    - Validate `content_hash` in `retrieve/exists/delete` against `^[0-9a-f]{64}$`; raise `ValueError` on invalid input.
    - In `retrieve()`: read bytes, recompute SHA-256, and raise if mismatch.
    - In `store()`: write to a temp file in the same directory and `replace()` atomically; if the destination exists, validate its hash and raise on mismatch.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests verifying:
    - corruption is detected on `retrieve()`
    - invalid hashes are rejected (including `..` / `/`)
    - atomic write doesn’t leave partial files (can be simulated by interrupt/mocking)
- Risks or migration steps:
  - Existing deployments with corrupted blobs will start failing fast (intended); provide a repair path (delete blob + keep hash) if policy allows.

## Architectural Deviations

- Spec or doc reference:
  - `docs/design/architecture.md:545-580` (payload_hash used for integrity verification; retention relies on hashes surviving deletion)
- Observed divergence:
  - Payload retrieval does not verify integrity and the path derivation does not enforce hash invariants.
- Alignment plan or decision needed:
  - Decide whether payload refs are strictly internal (Tier 1) or ever cross a trust boundary; regardless, fail-fast on invalid refs is consistent with audit DB “full trust” invariants.

## Acceptance Criteria

- `retrieve()` raises on any content hash mismatch.
- Invalid hashes are rejected before filesystem access.
- Writes are atomic and cannot leave partially-written blobs at the final hash path.
- New tests fail pre-fix and pass post-fix.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/test_payload_store.py`
- New tests required: yes (integrity + validation coverage)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md` (payload storage + retention)
