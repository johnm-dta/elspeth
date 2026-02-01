# Bug Report: Unvalidated content_hash enables path traversal

## Summary

- FilesystemPayloadStore constructs filesystem paths directly from `content_hash` without validating that it is a SHA-256 hex digest, allowing path traversal and non-hash strings to reach `exists()`/`delete()`/`retrieve()`.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- Path construction uses raw `content_hash` without validation: `src/elspeth/core/payload_store.py:39-42`.
- `retrieve()`, `exists()`, and `delete()` call `_path_for_hash()` directly with no validation: `src/elspeth/core/payload_store.py:56-90`.
- Contract explicitly states `content_hash` is a SHA-256 hex digest: `src/elspeth/contracts/payload_store.py:29-75`.
- Audit tier rules require crashing on invalid audit data: `CLAUDE.md:34-41`

## Impact

- User-facing impact: Potentially misleading behavior (payload reported missing/purged when the ref is malformed)
- Data integrity / security impact: Path traversal allows `exists()`/`delete()` to touch files outside the payload store; violates audit integrity and trust assumptions
- Performance or cost impact: Minor

## Root Cause Hypothesis

- Missing validation and containment checks for `content_hash` in `_path_for_hash()` and the public methods that call it.

## Proposed Fix

- Code changes (modules/files):
  - Add strict validation (length 64, lowercase hex) for `content_hash` in `src/elspeth/core/payload_store.py`
  - Enforce `base_path` containment (e.g., `resolve()` and parent check) before filesystem access
- Tests to add/update:
  - New unit tests in `tests/core/test_payload_store.py` to assert invalid hashes raise immediately and cannot traverse outside the base path

## Acceptance Criteria

- Invalid `content_hash` values raise immediately without touching the filesystem
- Path traversal attempts cannot access or delete files outside `base_path`
- Tests for invalid hashes/path traversal pass

## Verification (2026-02-01)

**Status: FIXED**

## Fix Implementation

**Changes made:**

1. **Added SHA-256 validation** in `_path_for_hash()`:
   - Compiled regex `^[a-f0-9]{64}$` validates exactly 64 lowercase hex characters
   - Raises `ValueError` with clear message on invalid format

2. **Added path containment check**:
   - After constructing path, resolves and verifies it's under `base_path`
   - Defense in depth against any edge cases the regex might miss

3. **Added security test suite** (`TestPayloadStoreSecurityValidation`):
   - `test_retrieve_rejects_path_traversal` - path traversal blocked
   - `test_exists_rejects_path_traversal` - cannot probe external files
   - `test_delete_rejects_path_traversal` - cannot delete external files
   - `test_rejects_non_hex_characters` - 'g', 'z', etc. rejected
   - `test_rejects_uppercase_hex` - 'A'-'F' rejected (must be lowercase)
   - `test_rejects_wrong_length` - too short/long rejected
   - `test_rejects_empty_hash` - empty string rejected
   - `test_accepts_valid_sha256_hash` - normal operation still works
   - `test_path_containment_after_resolution` - containment verified

4. **Updated existing tests** that used invalid hash formats:
   - `tests/core/test_payload_store.py` - changed `"nonexistent" * 4` to valid hex
   - `tests/core/checkpoint/test_recovery_row_data.py` - same fix

**Files modified:**
- `src/elspeth/core/payload_store.py` - added validation and containment
- `tests/core/test_payload_store.py` - added security tests, fixed existing tests
- `tests/core/checkpoint/test_recovery_row_data.py` - fixed test data

**Test results:**
- 21 payload store unit tests pass
- 9 payload store property tests pass
- 89 total payload-related tests pass
- mypy: no issues
- ruff: all checks pass

## Closure

- **Closed by:** Claude (systematic debugging fix)
- **Closure date:** 2026-02-01
- **Resolution:** Fixed with validation + containment + comprehensive tests
