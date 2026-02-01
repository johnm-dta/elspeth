# Bug Report: response_format config is ignored in Azure multi-query

## Summary

- AzureMultiQueryLLMTransform stores `response_format` but never sends it to the LLM API, so JSON enforcement is ignored and parsing failures increase.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any azure_multi_query_llm run with response_format set

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure `azure_multi_query_llm` with `response_format: json`.
2. Run a row and inspect the request sent to the LLM.

## Expected Behavior

- The LLM request includes response_format (or equivalent) to enforce JSON output when supported.

## Actual Behavior

- response_format is stored but never used in the chat_completion request.

## Evidence

- response_format is captured in `src/elspeth/plugins/llm/azure_multi_query.py:101`.
- chat_completion call omits response_format in `src/elspeth/plugins/llm/azure_multi_query.py:210`.

## Impact

- User-facing impact: higher rate of json_parse_failed errors.
- Data integrity / security impact: none direct, but more error routing.
- Performance or cost impact: wasted LLM calls due to parse failures.

## Root Cause Hypothesis

- response_format was added to config but not wired into the API call.

## Proposed Fix

- Code changes (modules/files): pass response_format to AuditedLLMClient.chat_completion (if supported) or remove the config field.
- Config or schema changes: document provider support for response_format.
- Tests to add/update:
  - Assert response_format is forwarded in the request payload.
- Risks or migration steps:
  - Ensure providers that do not support response_format handle it gracefully.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): response_format is a documented config field.
- Observed divergence: config is ignored.
- Reason (if known): incomplete wiring.
- Alignment plan or decision needed: confirm provider-specific parameter name.

## Acceptance Criteria

- response_format is included in LLM requests when configured (or removed if unsupported).

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_multi_query.py -v`
- New tests required: yes, request payload includes response_format.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: examples/multi_query_assessment/suite.yaml

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 4c

**Current Code Analysis:**

The bug remains present in the current codebase (commit 0a339fd). Examination of `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure_multi_query.py` confirms:

1. **Line 103**: `self._response_format = cfg.response_format` - The config value is stored in the instance variable
2. **Lines 221-226**: The `llm_client.chat_completion()` call only passes `model`, `messages`, `temperature`, and `max_tokens` - `response_format` is NOT included
3. The stored `_response_format` variable is never referenced anywhere else in the file

Additionally verified:
- The `AuditedLLMClient.chat_completion()` method (in `/home/john/elspeth-rapid/src/elspeth/plugins/clients/llm.py:119-226`) accepts `**kwargs` that are passed through to the underlying OpenAI client, so it COULD support `response_format` if passed
- The example config at `/home/john/elspeth-rapid/examples/multi_query_assessment/suite.yaml:113` explicitly sets `response_format: json`, expecting this feature to work
- The `MultiQueryConfig` dataclass includes `response_format` as a documented field with default value "json"

**Git History:**

No commits since the bug was reported (2026-01-21, commit ae2c0e6) have addressed this issue. The relevant commits to these files were:
- `0e2f6da` - Added validation to plugins (no response_format fix)
- `c786410` - RC-1 release (no response_format fix)
- Original implementation commits created the bug by storing but not using the field

**Root Cause Confirmed:**

YES - The bug is still present. The `response_format` configuration field is:
1. Accepted in config via `MultiQueryConfig.response_format`
2. Stored in `self._response_format` during `__init__`
3. Never passed to the LLM API call
4. Therefore has zero effect on LLM behavior

This means users configuring `response_format: json` to request JSON-mode responses from Azure OpenAI are not getting that enforcement, leading to higher parsing failure rates as the bug report correctly identified.

**Recommendation:**

**Keep open** - This is a valid P2 bug that should be fixed. The fix requires:

1. Passing `response_format` to `llm_client.chat_completion()` as a kwarg when it's configured
2. The OpenAI SDK expects `response_format={"type": "json_object"}` for JSON mode (not just the string "json"), so the code may need to transform the config value
3. Alternatively, if Azure OpenAI doesn't support this parameter in all API versions, the code should validate at init time or document the limitation
4. The example configuration in `examples/multi_query_assessment/suite.yaml` explicitly uses this feature, so users are expecting it to work

---

## RESOLUTION: 2026-01-26

**Status:** FIXED

**Fixed by:** Claude Code (fix/rc1-bug-burndown-session-5)

**Implementation:**
- Added conditional `response_format` parameter passing at line 211-224
- When `self._response_format == "json"`, passes `{"type": "json_object"}` to LLM API
- Uses proper OpenAI/Azure OpenAI format
- Reduces JSON parse failures by enforcing JSON mode at LLM level

**Code review:** Approved by pr-review-toolkit:code-reviewer agent

**Files changed:**
- `src/elspeth/plugins/llm/azure_multi_query.py`

### Code Evidence

**Before (lines 221-226 - response_format not passed):**
```python
response = llm_client.chat_completion(
    model=self._model,
    messages=messages,
    temperature=self._temperature,
    max_tokens=self._max_tokens,
    # ❌ response_format stored in self._response_format but never used
)
```

**After (lines 211-224 - response_format conditionally passed):**
```python
# Build kwargs for LLM call
llm_kwargs: dict[str, Any] = {
    "model": self._model,
    "messages": messages,
    "temperature": self._temperature,
    "max_tokens": self._max_tokens,
}

# ✅ Add response_format if configured (OpenAI expects {"type": "json_object"})
if self._response_format == "json":
    llm_kwargs["response_format"] = {"type": "json_object"}

try:
    response = llm_client.chat_completion(**llm_kwargs)  # ✅ Kwargs expanded
```

**Impact:**
- Config `response_format: json` now enforces JSON mode at LLM level
- Reduces JSON parse failures by instructing LLM to return valid JSON
- Uses proper OpenAI format: `{"type": "json_object"}`

**Verification:**
```bash
$ grep -n 'response_format.*json_object' src/elspeth/plugins/llm/azure_multi_query.py
221:            llm_kwargs["response_format"] = {"type": "json_object"}
```

Config value now properly passed to LLM API.
