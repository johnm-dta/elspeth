# Bug Report: Azure multi-query crashes if JSON response is not an object

## Summary

- After json.loads, AzureMultiQueryLLMTransform assumes a dict and indexes with string keys. If the LLM returns a JSON array or scalar, TypeError is raised and the transform crashes instead of returning a structured error.

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
- Data set or fixture: any azure_multi_query_llm run with non-object JSON response

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure `azure_multi_query_llm` normally.
2. Mock the LLM to return `[]` or `"ok"` as JSON.
3. Run a row.

## Expected Behavior

- Non-object JSON responses yield TransformResult.error with a clear reason.

## Actual Behavior

- TypeError occurs when checking membership or indexing parsed JSON, crashing the transform.

## Evidence

- Parsed JSON is used as a dict without type checks at `src/elspeth/plugins/llm/azure_multi_query.py:251` and `src/elspeth/plugins/llm/azure_multi_query.py:263`.

## Impact

- User-facing impact: rows fail with unhandled exceptions.
- Data integrity / security impact: missing structured error records.
- Performance or cost impact: retries/reruns needed.

## Root Cause Hypothesis

- Missing type validation for LLM JSON output.

## Proposed Fix

- Code changes (modules/files): validate `parsed` is a dict before mapping fields; otherwise return TransformResult.error with raw_response.
- Config or schema changes: N/A
- Tests to add/update:
  - Add tests for list/scalar JSON responses.
- Risks or migration steps:
  - None.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): external data parsing should be wrapped.
- Observed divergence: external response can crash parsing.
- Reason (if known): missing guardrails.
- Alignment plan or decision needed: standardize JSON response validation.

## Acceptance Criteria

- Non-object JSON responses yield structured TransformResult.error without exceptions.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_multi_query.py -v`
- New tests required: yes, non-object JSON response handling.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md trust model
