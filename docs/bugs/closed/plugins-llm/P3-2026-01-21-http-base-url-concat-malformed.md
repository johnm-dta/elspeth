# Bug Report: AuditedHTTPClient base_url concatenation can create malformed URLs

## Summary

- `AuditedHTTPClient` concatenates `base_url` and `url` via string interpolation. Missing or extra slashes yield malformed URLs (e.g., `.../v1process` or double slashes), and absolute URLs get incorrectly prefixed.

## Severity

- Severity: minor
- Priority: P3

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
- Notable tool calls or steps: code inspection of URL handling

## Steps To Reproduce

1. Configure `base_url="https://api.example.com/v1"` and call `post("process", json={...})`.
2. Observe full URL becomes `https://api.example.com/v1process`.
3. Configure `base_url="https://api.example.com/"` and call `post("/v1/process", ...)`.
4. Observe double slash in URL (`https://api.example.com//v1/process`).

## Expected Behavior

- URL joining should be robust to leading/trailing slashes and absolute URLs.

## Actual Behavior

- URLs are concatenated naively, producing malformed or unintended endpoints.

## Evidence

- String concatenation of base URL and path: `src/elspeth/plugins/clients/http.py:134`

## Impact

- User-facing impact: requests can target wrong endpoints or fail with malformed URL errors.
- Data integrity / security impact: low.
- Performance or cost impact: low.

## Root Cause Hypothesis

- URL construction uses string concatenation instead of a proper URL join utility.

## Proposed Fix

- Code changes (modules/files):
  - Use `httpx.URL` or `urllib.parse.urljoin`, or configure `httpx.Client(base_url=...)` and pass relative paths.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests for base_url joining with and without leading/trailing slashes.
- Risks or migration steps:
  - Ensure existing callers that pass full URLs continue to work.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: N/A
- Reason (if known): simple concatenation.
- Alignment plan or decision needed: N/A

## Acceptance Criteria

- `AuditedHTTPClient` produces correct URLs for common `base_url` and `url` combinations.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k base_url`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Current Code Analysis

The bug still exists in `/home/john/elspeth-rapid/src/elspeth/plugins/clients/http.py` at line 134:

```python
full_url = f"{self._base_url}{url}" if self._base_url else url
```

This naive string concatenation produces malformed URLs in multiple scenarios.

### Reproduction Confirmed

Tested both failure scenarios described in the original report:

**Scenario 1 - Missing Slash:**
```python
base_url = "https://api.example.com/v1"
url = "process"
result = "https://api.example.com/v1process"  # BUG: Missing separator
expected = "https://api.example.com/v1/process"
```

**Scenario 2 - Double Slash:**
```python
base_url = "https://api.example.com/"
url = "/v1/process"
result = "https://api.example.com//v1/process"  # BUG: Double slash
expected = "https://api.example.com/v1/process"
```

**Existing Test (Passes by Accident):**
The current test at `tests/plugins/clients/test_audited_http_client.py:167-198` only tests:
```python
base_url = "https://api.example.com"  # No trailing slash
url = "/v1/process"                    # Leading slash
result = "https://api.example.com/v1/process"  # Works correctly
```

This test passes because the leading slash in `url` provides the separator, masking the underlying concatenation bug.

### Git History Review

No fixes found since bug report date (2026-01-21). Recent commits to `http.py`:
- `b9a7bfa` (2026-01-20): Record full HTTP response body (unrelated)
- `15b78a4`: Filter sensitive response headers (unrelated)
- `242ed66`: Initial audited client infrastructure (unrelated)

### Test Coverage Gap

The existing `test_base_url_prepended` test does not cover:
1. Base URL with trailing slash + URL with leading slash (double slash)
2. Base URL without trailing slash + URL without leading slash (missing slash)
3. Absolute URLs being incorrectly prefixed

### Solution Options

1. **httpx.Client with base_url** (Recommended): Configure `httpx.Client(base_url=...)` and let httpx handle URL joining
2. **httpx.URL.join()**: Use `httpx.URL(base_url).join(url)` (RFC 3986 semantics)
3. **urllib.parse.urljoin**: Standard library URL joining

**Note on httpx.URL.join()**: Testing shows it follows RFC 3986 semantics where leading `/` means "absolute path" (replaces entire path component), which may not be the desired behavior. For example:
```python
httpx.URL("https://api.example.com/v1").join("/process")
# Returns: https://api.example.com/process (v1 is replaced)
```

Using `httpx.Client(base_url=...)` is likely the best approach as it's designed for this use case.

### Impact Assessment

- **Current Risk**: Medium - callers must know exact slash conventions to avoid malformed URLs
- **Workaround**: Callers can pass full URLs (no base_url) or carefully manage slashes
- **Fix Priority**: P3 is appropriate - real issue but has workarounds and hasn't caused production failures

### Recommendation

Bug remains valid and should be fixed. Recommend:
1. Switch to `httpx.Client(base_url=...)` pattern
2. Add comprehensive URL joining tests covering all slash combinations
3. Document expected URL format in docstring (relative vs absolute paths)

## Resolution

**Fixed in:** 2026-01-29
**Fix:** Replaced naive string concatenation with proper slash normalization. The new code strips trailing slashes from base URL and leading slashes from path, then joins with exactly one slash.

**Changes:**
- `src/elspeth/plugins/clients/http.py`: Updated URL joining logic at lines 192-200
- `tests/plugins/clients/test_audited_http_client.py`: Added edge case tests for slash handling
