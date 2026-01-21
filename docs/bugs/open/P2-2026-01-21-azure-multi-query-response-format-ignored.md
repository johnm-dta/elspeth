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
