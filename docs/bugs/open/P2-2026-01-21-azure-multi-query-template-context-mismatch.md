# Bug Report: Multi-query templates cannot use input_1/criterion at top level

## Summary

- PromptTemplate only exposes `row` and `lookup`, but multi-query configs/examples use `{{ input_1 }}` and `{{ criterion.name }}` at top-level. Rendering will raise TemplateError for those templates, causing every query to fail.

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
- Data set or fixture: examples/multi_query_assessment/suite.yaml

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Run `azure_multi_query_llm` with a template using `{{ input_1 }}` or `{{ criterion.name }}` (as in examples).
2. Execute a row.
3. Observe TemplateError: undefined variable.

## Expected Behavior

- Templates can use `input_1`, `criterion`, and `case_study` at the top level, matching examples and docs.

## Actual Behavior

- PromptTemplate only provides `row` and `lookup`; top-level variables are undefined.

## Evidence

- PromptTemplate context is fixed to `row` and `lookup` in `src/elspeth/plugins/llm/templates.py:148`.
- Multi-query uses PromptTemplate with a synthetic row in `src/elspeth/plugins/llm/azure_multi_query.py:182`.
- Template context is built with `input_1`, `criterion`, and `case_study` in `src/elspeth/plugins/llm/multi_query.py:47` but is nested under `row` at render time.

## Impact

- User-facing impact: multi-query templates in examples fail immediately.
- Data integrity / security impact: rows routed to on_error or failed.
- Performance or cost impact: wasted LLM calls if template errors are not caught early.

## Root Cause Hypothesis

- PromptTemplate enforces a single `row` namespace, but multi-query design expects top-level variables.

## Proposed Fix

- Code changes (modules/files):
  - Either render multi-query templates with a custom environment that exposes `input_1`, `criterion`, `case_study` at top level, or adjust the template contract and update examples to `{{ row.input_1 }}` and `{{ row.criterion.name }}`.
- Config or schema changes: N/A
- Tests to add/update:
  - Add a test that renders a template using top-level `input_1` and `criterion`.
- Risks or migration steps:
  - Decide on a stable template contract and update existing configs/examples accordingly.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): examples/multi_query_assessment and tests/contracts expect top-level variables.
- Observed divergence: runtime only supports row.* namespace.
- Reason (if known): PromptTemplate API reuse.
- Alignment plan or decision needed: choose template namespace contract for multi-query.

## Acceptance Criteria

- Multi-query templates using `input_1`/`criterion` render successfully and produce prompts.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_multi_query.py -v`
- New tests required: yes, template contract test with top-level variables.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: examples/multi_query_assessment/suite.yaml
