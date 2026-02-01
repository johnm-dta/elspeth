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

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID

- Empty batch path still returns `payload_size=0` while the hash is computed from canonical JSON (`"[]"`). (`src/elspeth/plugins/sinks/database_sink.py:281-297`)

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 3

**Current Code Analysis:**

The bug is still present in the current codebase at `/home/john/elspeth-rapid/src/elspeth/plugins/sinks/database_sink.py`.

Lines 215-229 show the exact issue:

```python
# Lines 215-219: Compute canonical JSON payload BEFORE any database operation
payload_json = json.dumps(rows, sort_keys=True, separators=(",", ":"))
payload_bytes = payload_json.encode("utf-8")
content_hash = hashlib.sha256(payload_bytes).hexdigest()
payload_size = len(payload_bytes)  # This is 2 for "[]"

# Lines 221-229: Empty batch handling
if not rows:
    # Empty batch - return descriptor without DB operations
    return ArtifactDescriptor.for_database(
        url=self._sanitized_url,
        table=self._table_name,
        content_hash=content_hash,
        payload_size=0,  # BUG: Hardcoded to 0, should be payload_size (which is 2)
        row_count=0,
    )
```

The code correctly computes `payload_size = len(payload_bytes)` which evaluates to `2` for the JSON string `"[]"`. However, when returning the ArtifactDescriptor for empty batches, it hardcodes `payload_size=0` instead of using the computed variable.

**Git History:**

Searched commits since 2026-01-21:
- `7ee7c51` - feat: add self-validation to all builtin plugins (not related)
- `dd3bed7` - fix(security): honor dev-mode override in DatabaseSink (not related)
- `57c57f5` - fix: resolve 8 RC1 bugs (fixed different DatabaseSink bug about if_exists="replace")

None of these commits addressed the payload_size inconsistency.

**Root Cause Confirmed:**

Yes, the bug is still present. The inconsistency is:

1. `content_hash = hashlib.sha256(payload_bytes).hexdigest()` hashes the full JSON payload `"[]"` (2 bytes)
2. `payload_size=0` is hardcoded in the return statement instead of using the computed `payload_size` variable

This creates an internal inconsistency where the hash represents 2 bytes of content but the size field claims 0 bytes.

**Test Evidence:**

The bug is codified in the test at `/home/john/elspeth-rapid/tests/plugins/sinks/test_database_sink.py:153-168`:

```python
def test_batch_write_empty_list(self, db_url: str, ctx: PluginContext) -> None:
    """Batch write with empty list returns descriptor with zero size."""
    # ...
    artifact = sink.write([], ctx)

    assert artifact.size_bytes == 0  # Test expects buggy behavior
    # Empty payload hash
    empty_json = json.dumps([], sort_keys=True, separators=(",", ":"))
    assert artifact.content_hash == hashlib.sha256(empty_json.encode()).hexdigest()
```

The test explicitly asserts `size_bytes == 0` while also verifying the hash matches `"[]"`, proving the inconsistency exists and is tested for.

**Recommendation:**

Keep open. This is a valid P3 bug that violates the internal consistency contract between `content_hash` and `size_bytes`. The fix is trivial: change line 227 from `payload_size=0` to `payload_size=payload_size` (or just `payload_size` for brevity). The corresponding test assertion at line 164 should be updated to `assert artifact.size_bytes == 2` to verify the fix.

## Closure Report (2026-02-01)

**Status:** CLOSED (IMPLEMENTED)

### Fix Summary

- Empty batch now returns `payload_size` matching canonical JSON (`"[]"`).

### Test Coverage

- `tests/plugins/sinks/test_database_sink.py::TestDatabaseSink::test_batch_write_empty_list`
