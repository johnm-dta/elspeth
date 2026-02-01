# Bug Report: AuditedLLMClient records only partial LLM responses

## Summary

- `AuditedLLMClient` records only `content`, `model`, and `usage` in `response_data`, discarding full response details (additional choices, tool calls, finish reasons, logprobs). The raw response is returned to the caller but not stored in the audit trail, violating full response recording requirements and limiting replay/verify.

## Severity

- Severity: major
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
- Notable tool calls or steps: code inspection of audited LLM client recording

## Steps To Reproduce

1. Call `AuditedLLMClient.chat_completion` with `n>1` or tool call outputs enabled.
2. Inspect the recorded call in the audit trail.
3. Observe only a single `content` string and minimal metadata are stored.

## Expected Behavior

- The full LLM response (all choices, tool calls, finish reasons, etc.) should be recorded in the audit trail.

## Actual Behavior

- Only a subset of the response is stored; raw response data is lost in the audit record.

## Evidence

- Partial response recording: `src/elspeth/plugins/clients/llm.py:181-191`
- Raw response kept only in return value: `src/elspeth/plugins/clients/llm.py:195-200`

## Impact

- User-facing impact: replay/verify cannot reproduce full model outputs (tool calls, multiple choices).
- Data integrity / security impact: audit trail is incomplete for external calls.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Implementation records only derived fields instead of the full provider response.

## Proposed Fix

- Code changes (modules/files):
  - Record the full response payload (e.g., `response.model_dump()` or provider-equivalent) in `response_data` or payload store.
  - Keep summary fields (`content`, `usage`) for convenience, but do not drop full response data.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests that assert tool call responses and multiple choices are preserved in recorded payloads.
- Risks or migration steps:
  - Ensure canonicalization can handle the full response structure.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (auditability standard: "External calls - Full request AND response recorded")
- Observed divergence: LLM responses are partially recorded.
- Reason (if known): convenience and reduced payload size.
- Alignment plan or decision needed: define storage requirements for full LLM responses.

## Acceptance Criteria

- Recorded LLM calls contain the complete provider response, including tool calls and multi-choice data.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k llm_response`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Code Inspection

The bug remains present in the current codebase. Inspection of `/home/john/elspeth-rapid/src/elspeth/plugins/clients/llm.py` confirms:

**Lines 187-191** - Only partial response data is recorded to audit trail:
```python
response_data={
    "content": content,           # Only first choice content
    "model": response.model,      # Model name
    "usage": usage,               # Token counts only
},
```

**Line 200** - Full response is captured but NOT stored in audit trail:
```python
raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
```

The `raw_response` field is included in the returned `LLMResponse` object for debugging purposes, but it is NOT passed to `recorder.record_call()` and therefore NOT stored in the audit trail.

### Missing Response Fields

The following OpenAI API response fields are being discarded:

1. **Multiple choices**: Only `choices[0]` content is extracted; if `n>1` parameter is used, additional choices are lost
2. **Finish reasons**: `choices[i].finish_reason` (e.g., "stop", "length", "content_filter", "tool_calls") not recorded
3. **Tool calls**: `choices[i].message.tool_calls` not recorded (critical for function calling)
4. **Logprobs**: `choices[i].logprobs` not recorded (important for confidence scoring)
5. **Response metadata**: Response ID, created timestamp, system fingerprint not recorded
6. **Message role**: `choices[i].message.role` not recorded

### Current Usage Patterns

Codebase search reveals no current usage of:
- `n>1` parameter for multiple completions
- Tool calls (`tool_calls` field)
- Finish reasons (`finish_reason` field)

However, this is a **latent bug** - the system architecture violates the stated auditability standard even if these features aren't currently used.

### Git History

No commits since the bug was reported (2026-01-21) have addressed this issue. The last change to `llm.py` was the RC-1 release commit (c786410) which did not modify the response recording logic.

### Architectural Violation

This violates the explicit CLAUDE.md requirement:

> **Data storage points** (non-negotiable):
> 3. **External calls** - Full request AND response recorded

The current implementation records the full request but only a partial response.

### Payload Store Not Used

The code does not utilize the payload store mechanism (`request_ref`/`response_ref` parameters in `record_call()`) which could store the full response separately. The LLM client directly passes the partial `response_data` dict.

### Testing Gap

Tests in `/home/john/elspeth-rapid/tests/plugins/clients/test_audited_llm_client.py` only verify that the partial fields are recorded. There are no tests asserting that:
- Multiple choices are preserved
- Tool calls are recorded
- Finish reasons are captured
- Full response can be retrieved from audit trail

### Impact Assessment

**Current Impact**: Low - No current code uses advanced LLM features (tool calls, multiple choices)

**Future Impact**: High - Any future implementation of:
- Function calling / tool use
- Multi-choice sampling for quality
- Content filtering detection
- Response verification/replay

...would be unable to reconstruct what the LLM actually returned.

### Recommendation

Bug should remain **P1** priority. While not currently causing failures, it represents a fundamental architectural gap that blocks important future capabilities and violates the auditability contract.

---

## Resolution (2026-01-28)

**Status: FIXED**

### Fix Summary

Added `raw_response` field to the `response_data` dict passed to `recorder.record_call()`. The full response from `response.model_dump()` is now recorded in the audit trail alongside the summary fields.

### Code Changes

**File:** `src/elspeth/plugins/clients/llm.py`

Changed `record_call()` invocation (lines 297-316) to include full raw response:

```python
# Capture full raw response for audit completeness
# raw_response includes: all choices, finish_reason, tool_calls, logprobs, etc.
raw_response = response.model_dump() if hasattr(response, "model_dump") else None

self._recorder.record_call(
    state_id=self._state_id,
    call_index=call_index,
    call_type=CallType.LLM,
    status=CallStatus.SUCCESS,
    request_data=request_data,
    response_data={
        # Summary fields for convenience
        "content": content,
        "model": response.model,
        "usage": usage,
        # Full response for audit completeness (tool_calls, multiple choices, etc.)
        "raw_response": raw_response,
    },
    latency_ms=latency_ms,
)
```

### Tests Added

**File:** `tests/plugins/clients/test_audited_llm_client.py`

Added 4 new tests:
1. `test_full_raw_response_recorded_in_audit_trail` - Verifies complete response structure preserved
2. `test_multiple_choices_preserved_in_raw_response` - Verifies n>1 choices captured
3. `test_tool_calls_preserved_in_raw_response` - Verifies function calling data captured
4. `test_raw_response_none_when_model_dump_unavailable` - Handles edge case gracefully

### Verification

- All 21 tests in `test_audited_llm_client.py` pass
- All 449 tests in `tests/plugins/clients/` and `tests/plugins/llm/` pass
- mypy type checking passes
- ruff linting passes

### Architectural Compliance

This fix restores compliance with CLAUDE.md:

> **Data storage points** (non-negotiable):
> 3. **External calls** - Full request AND response recorded

The audit trail now contains the complete LLM provider response, enabling:
- Replay/verify mode for full response reconstruction
- Tool call auditing for function calling
- Multi-choice analysis when n>1
- Finish reason tracking (stop, length, content_filter, tool_calls)
- Response metadata (id, system_fingerprint, created timestamp)
