# Bug Report: AuditedHTTPClient truncates non-JSON responses in audit trail

## Summary

- For non-JSON responses, `AuditedHTTPClient` truncates the response body to 100,000 characters before recording. This violates the "full response recorded" audit requirement and makes replay/verify impossible for large or binary payloads.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/clients` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of audited HTTP client response handling

## Steps To Reproduce

1. Call an endpoint that returns a large non-JSON body (>100k bytes).
2. Observe the recorded response body in the audit trail is truncated to 100,000 characters.
3. Attempt replay/verify; the recorded response does not match the original payload.

## Expected Behavior

- The full response payload should be recorded (via payload store if necessary), preserving the complete body and hash.

## Actual Behavior

- Response bodies are truncated for non-JSON content, losing data.

## Evidence

- Truncation logic: `src/elspeth/plugins/clients/http.py:168-170`

## Impact

- User-facing impact: replay/verify cannot reproduce or validate large or binary responses.
- Data integrity / security impact: audit trail is incomplete; hashes refer to truncated payloads.
- Performance or cost impact: unclear; truncation may be hiding a need for proper payload storage.

## Root Cause Hypothesis

- Non-JSON responses are coerced to text and truncated instead of being stored as full payload bytes.

## Proposed Fix

- Code changes (modules/files):
  - Store full response bytes in the payload store and record metadata (size, content type) in `response_data`.
  - Consider base64 encoding for binary payloads if structured storage is required.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that records a large non-JSON response and asserts the full payload is recoverable.
- Risks or migration steps:
  - Ensure payload retention policies can handle large responses; document size implications.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (auditability standard: "External calls - Full request AND response recorded")
- Observed divergence: non-JSON responses are truncated.
- Reason (if known): size guardrail.
- Alignment plan or decision needed: decide on payload retention strategy for large responses.

## Acceptance Criteria

- Recorded HTTP responses preserve the full payload (or a recoverable full payload via payload store) regardless of size or content type.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k http_response`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Current Code Analysis

The truncation bug is confirmed in the current codebase at `/home/john/elspeth-rapid/src/elspeth/plugins/clients/http.py:169-170`:

```python
else:
    # For non-JSON, store text (truncated for very large responses)
    response_body = response.text[:100_000] if len(response.text) > 100_000 else response.text
```

### Key Findings

1. **Truncation is Active**: Non-JSON responses larger than 100,000 characters are truncated before being passed to `record_call()`.

2. **Infrastructure Exists**: The payload store infrastructure is already in place and working:
   - `LandscapeRecorder.record_call()` supports `response_ref` parameter for payload store references
   - Lines 2073-2077 in `recorder.py` show automatic payload persistence: if a payload_store is available, `record_call()` will automatically persist response_data to the store
   - The comment in commit `b9a7bfa` claims "The payload store (via record_call auto-persist) handles large bodies" but the code contradicts this by truncating before calling `record_call()`

3. **Git History**: The truncation was introduced in commit `b9a7bfa` (2026-01-20) which added response body recording. The commit message incorrectly states the payload store handles large bodies, but the implementation truncates before the payload store can see the full response.

4. **No Tests**: There are no tests for large responses (>100KB) in `tests/plugins/clients/test_audited_http_client.py`. Tests exist for JSON and non-JSON responses but only with small payloads.

5. **Binary Response Issue**: The code calls `response.text` for all non-JSON responses (line 170), which will fail for binary content (images, PDFs, etc.) since `.text` attempts UTF-8 decoding. Binary responses need `response.content` (bytes) instead.

### Impact Assessment

- **Audit Integrity Violation**: Hashes stored in the audit trail reference truncated payloads, not the actual responses received
- **Replay/Verify Broken**: Cannot reproduce or validate large responses from the audit trail
- **Binary Support Missing**: Binary responses will either fail (decode error) or be corrupted if they happen to decode as UTF-8
- **False Safety**: The truncation creates the illusion of audit compliance while actually losing data

### Fix Requirements

The fix needs to:
1. Remove the truncation at line 170
2. Handle binary responses by storing `response.content` (bytes) instead of `response.text` for non-JSON responses
3. Let the payload store auto-persist mechanism (already implemented in `recorder.py`) handle large payloads
4. Add test coverage for:
   - Large text responses (>100KB)
   - Large JSON responses (>100KB)
   - Binary responses (images, PDFs)
   - Verify full payload is recoverable from audit trail

### Architectural Notes

The payload store pattern is correctly implemented in `LandscapeRecorder` - the HTTP client just needs to trust it and pass full responses instead of pre-truncating them. The truncation defeats the entire purpose of the payload store architecture.

## Resolution (2026-01-26)

**Status: FIXED**

### Root Cause

HTTP client truncated non-JSON responses to 100,000 characters before passing them to `record_call()`, which defeated the payload store auto-persist mechanism already implemented in `LandscapeRecorder` (lines 2216-2226 in `recorder.py`).

The commit message for `b9a7bfa` claimed "The payload store (via record_call auto-persist) handles large bodies" but the implementation truncated responses before the payload store could see them - indicating incomplete understanding of the payload store architecture during implementation.

**Secondary issue**: Binary responses (images, PDFs) would fail because code called `response.text` which attempts UTF-8 decoding on binary data.

### Fix Applied

**Removed truncation completely** - let payload store auto-persist handle large responses as designed.

**Added content-type detection** to distinguish:
- JSON responses: parsed as `dict` (unchanged)
- Text responses: stored as full `str` (no truncation)
- Binary responses: base64-encoded as `{"_binary": "<base64>"}` for JSON serialization

**Pattern alignment**: LLM client already used this pattern (no truncation, trust payload store). HTTP client now follows the same approach.

### Changes

**Modified files:**
- `src/elspeth/plugins/clients/http.py` (lines 159-185)
  - Added `import base64`
  - Removed truncation at 100KB limit
  - Added content-type detection (text vs binary)
  - Binary responses encoded as base64 for canonical JSON

- `tests/plugins/clients/test_audited_http_client.py` (added 3 tests)
  - `test_large_text_response_not_truncated` - 150KB text response
  - `test_large_json_response_not_truncated` - >100KB JSON response
  - `test_binary_response_recorded_as_base64` - PNG image response

### Test Results

**Before fix:**
- ❌ Large text test FAILED - truncated to 100,000 chars
- ✅ Large JSON test PASSED - bug only affected non-JSON
- ❌ Binary test FAILED - called `.text` on binary data

**After fix:**
- ✅ All 3 new tests PASS
- ✅ All 23 HTTP client tests PASS (no regressions)
- ✅ Full test suite: 3,563 passed, 0 failures

### Commit

- Hash: (pending)
- Message: "fix(http): remove response truncation, add binary support"
