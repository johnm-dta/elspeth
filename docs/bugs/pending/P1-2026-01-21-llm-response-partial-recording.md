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
