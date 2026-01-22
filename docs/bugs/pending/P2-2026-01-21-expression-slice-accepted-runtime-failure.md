# Bug Report: ExpressionParser accepts slicing but evaluator treats slice as None

## Summary

- Slice syntax (e.g., `row['items'][1:3]`) passes validation but fails at runtime because `_ExpressionEvaluator` does not handle `ast.Slice`, so the slice evaluates to `None` and indexing crashes.

## Severity

- Severity: major
- Priority: P2

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

- Goal or task prompt: deep dive into `src/elspeth/engine/expression_parser.py` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of validator/evaluator slice handling

## Steps To Reproduce

1. Create an expression like `row['items'][1:3] == [1, 2]`.
2. Parse it with `ExpressionParser` (validation passes).
3. Evaluate against `{"items": [0, 1, 2, 3]}`.
4. Observe a `TypeError` or `KeyError` because the slice evaluates to `None` and `value[None]` is attempted.

## Expected Behavior

- Slice syntax should either be explicitly rejected at validation time or evaluated correctly (using `slice(lower, upper, step)`), but it should not pass validation and then crash at runtime.

## Actual Behavior

- `ast.Slice` nodes are not handled; `self.visit(node.slice)` returns `None`, and the evaluator attempts `value[None]`.

## Evidence

- Subscript validation allows any slice: `src/elspeth/engine/expression_parser.py:85-88`
- Evaluator uses `self.visit(node.slice)` without `ast.Slice` handling: `src/elspeth/engine/expression_parser.py:253-257`

## Impact

- User-facing impact: config conditions with slices crash pipeline execution.
- Data integrity / security impact: low.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `ast.Slice` is neither forbidden in validation nor implemented in the evaluator.

## Proposed Fix

- Code changes (modules/files):
  - Either explicitly reject `ast.Slice` / `ast.ExtSlice` in `_ExpressionValidator`, or add evaluator support to translate slices into `slice()`.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test asserting slice expressions are rejected (or evaluated correctly if supported).
- Risks or migration steps:
  - Decide whether slicing should be part of the allowed expression subset.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (allowed operations list does not include slicing)
- Observed divergence: slice syntax is accepted by validation but not supported in evaluation.
- Reason (if known): missing AST handler for `ast.Slice`.
- Alignment plan or decision needed: clarify whether slicing should be allowed.

## Acceptance Criteria

- Slice expressions are either rejected at parse time or evaluated correctly without runtime exceptions.

## Tests

- Suggested tests to run: `pytest tests/engine/test_expression_parser.py -k slice`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
