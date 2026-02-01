# Bug Report: LLM response payload dropped when post-call parsing fails

## Summary

- When LLM response parsing fails (e.g., empty choices, model_dump error), the exception handler records an error but does NOT record the actual response payload. This violates "External calls - Full request AND response recorded".

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/clients/llm.py:287-369`:
  - Entire call + response parsing in one try block
  - Line 291 `response.choices[0].message.content` can fail (empty choices)
  - Line 304 `response.model_dump()` can fail
  - Error handler at lines 339-351 records `error=` but no `response_data=`
- A response object existed but is lost when parsing fails

## Impact

- User-facing impact: Cannot debug LLM failures without seeing actual response
- Data integrity / security impact: Audit trail incomplete for external calls
- Performance or cost impact: Wasted investigation time

## Root Cause Hypothesis

- Response parsing happens before recording, so failures lose the response. Should capture response immediately after HTTP call, then parse.

## Proposed Fix

- Code changes:
  - Capture `response_data = response.model_dump()` in inner try immediately after call
  - Parse content in separate try block
  - Always record response_data (success or failure)
- Tests to add/update:
  - Add test with empty choices response, verify response_data is still recorded

## Acceptance Criteria

- LLM response payloads are always recorded, even when parsing fails
- Error records include both error info AND the raw response
