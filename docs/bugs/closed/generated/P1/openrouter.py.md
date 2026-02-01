# Bug Report: Unhandled canonical_json errors during template rendering crash OpenRouter transform

## Summary

- OpenRouterLLMTransform only catches TemplateError around render_with_metadata; ValueError/TypeError from canonical_json (e.g., NaN/Infinity in row data) escapes and crashes the pipeline instead of returning TransformResult.error for row-level quarantine.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 86357898ee109a1dbb8d60f3dc687983fa22c1f0 (fix/rc1-bug-burndown-session-4)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Row with NaN/Infinity (e.g., `{"text": "x", "value": numpy.nan}`) under dynamic schema

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of `src/elspeth/plugins/llm/openrouter.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure `openrouter_llm` with `schema: {fields: dynamic}` and any template, then feed a row containing `NaN`/`Infinity` (or another canonical_json-rejected value).
2. Run the pipeline so the OpenRouter transform processes that row (or call `OpenRouterLLMTransform.process` with a valid `PluginContext`).

## Expected Behavior

- The transform should catch row-data hashing failures, return `TransformResult.error(...)`, and allow the engine to route/quarantine the row via `on_error` without crashing the run.

## Actual Behavior

- `canonical_json(row)` raises `ValueError` inside `render_with_metadata`, which is not caught, causing an exception to propagate and crash the transform/pipeline.

## Evidence

- `OpenRouterLLMTransform` only catches `TemplateError` around `render_with_metadata` in all paths: `src/elspeth/plugins/llm/openrouter.py:203-214`, `src/elspeth/plugins/llm/openrouter.py:403-415`, `src/elspeth/plugins/llm/openrouter.py:563-574`.
- `PromptTemplate.render_with_metadata` calls `canonical_json(row)` without handling its exceptions: `src/elspeth/plugins/llm/templates.py:163-176`.
- `canonical_json` explicitly raises `ValueError` for NaN/Infinity: `src/elspeth/core/canonical.py:57-60`, `src/elspeth/core/canonical.py:97-100`.

## Impact

- User-facing impact: A single malformed row can crash the entire run instead of being quarantined.
- Data integrity / security impact: Violates Tier-2 handling rules; audit trail may end abruptly without row-level error routing.
- Performance or cost impact: Run termination leads to re-runs and wasted compute/API calls.

## Root Cause Hypothesis

- Error handling around template rendering is too narrow; it assumes only `TemplateError` can occur, but `render_with_metadata` can raise `ValueError/TypeError` from canonicalization of row data.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/llm/openrouter.py`: broaden the `render_with_metadata` try/except to catch `ValueError` and `TypeError` (and possibly wrap them into a structured error reason) in `_process_sequential`, `process` (sequential path), and `_process_single_with_state`.
- Config or schema changes: Unknown.
- Tests to add/update:
  - Add a test in `tests/plugins/llm/test_openrouter.py` that passes a row with `numpy.nan` (or `float("inf")`) and asserts a `TransformResult.error` is returned rather than an exception.
- Risks or migration steps:
  - Low risk; ensure the catch scope is limited to row-data canonicalization failures to avoid masking internal bugs.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:120-127`
- Observed divergence: Row-data operations (canonicalization for template metadata) are not wrapped, violating the “operating on row field values? wrap, return error result, quarantine row” rule.
- Reason (if known): Unknown.
- Alignment plan or decision needed: Catch canonicalization errors in OpenRouter’s template rendering path and return `TransformResult.error`.

## Acceptance Criteria

- Row containing NaN/Infinity no longer crashes the run; OpenRouter returns `TransformResult.error` with a clear reason.
- Engine routes the error via `on_error` as configured and continues processing other rows.
- Unit test for NaN/Infinity in OpenRouter passes.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_openrouter.py`
- New tests required: yes, add a case covering NaN/Infinity row data in template rendering.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
