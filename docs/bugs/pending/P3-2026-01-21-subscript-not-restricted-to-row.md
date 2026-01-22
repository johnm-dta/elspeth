# Bug Report: Subscript access is not restricted to row data

## Summary

- The validator allows subscript access on any expression, and the evaluator does not enforce that the base object is `row` (or derived row data). This expands the allowed expression language beyond the documented `row['field']` access and can invoke `__getitem__` on arbitrary objects stored in rows.

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

- Goal or task prompt: deep dive into `src/elspeth/engine/expression_parser.py` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of subscript validation and evaluation

## Steps To Reproduce

1. Use an expression like `{'a': 1}['a'] == 1` or `'abc'[0] == 'a'`.
2. `ExpressionParser` accepts the expression even though it is not `row[...]` access.
3. The evaluator executes the subscript on the literal.

## Expected Behavior

- Subscript access should be restricted to `row[...]` (and nested structures within row data) per the documented expression language and security model.

## Actual Behavior

- Any subscript expression is allowed, and the evaluator performs `value[key]` without checking that `value` is `row` or safe.

## Evidence

- Validator allows all Subscript nodes with a comment that evaluator will enforce row access: `src/elspeth/engine/expression_parser.py:85-88`
- Evaluator performs unchecked indexing: `src/elspeth/engine/expression_parser.py:253-257`

## Impact

- User-facing impact: expressions can reference data outside the documented subset, leading to unexpected behavior.
- Data integrity / security impact: potential to trigger `__getitem__` side effects on non-primitive row objects.
- Performance or cost impact: low.

## Root Cause Hypothesis

- Subscript validation is overly permissive and the evaluator does not enforce a row-only base.

## Proposed Fix

- Code changes (modules/files):
  - Restrict subscript base to `row` or to values derived from `row` in the evaluator, or tighten validation to ensure the base expression is `row[...]`/`row.get(...)`.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests that reject subscript on literals (e.g., `"abc"[0]`) unless explicitly allowed.
- Risks or migration steps:
  - Ensure nested access like `row['data']['nested']` remains supported if intended.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (allowed field access is `row['field']`, `row.get('field')`)
- Observed divergence: subscript access applies to any expression.
- Reason (if known): evaluator never enforced row-only access.
- Alignment plan or decision needed: decide whether literal subscripts should be allowed.

## Acceptance Criteria

- Subscript access is limited to the intended row data scope, and unsupported subscripts are rejected at validation time.

## Tests

- Suggested tests to run: `pytest tests/engine/test_expression_parser.py -k subscript`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
