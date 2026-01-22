# Bug Report: is_boolean_expression misclassifies and/or expressions that return non-booleans

## Summary

- `ExpressionParser.is_boolean_expression()` returns True for all `and/or` expressions, even though evaluation follows Python semantics and can return non-boolean values. This causes config validation to require `true/false` route labels for expressions that actually return strings or numbers, leading to runtime routing failures or false config errors.

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
- Notable tool calls or steps: code inspection of boolean classification and evaluation

## Steps To Reproduce

1. Configure a gate condition like `row.get('label') or 'unknown'`.
2. Define routes that match string labels (e.g., `{"vip": "continue", "unknown": "review"}`).
3. Config validation calls `is_boolean_expression()` and incorrectly requires `true/false` labels, rejecting the config.
4. Alternatively, use true/false labels and observe runtime failure when the expression returns a string (route label not found).

## Expected Behavior

- `is_boolean_expression()` should only return True when the expression is guaranteed to evaluate to a boolean. For `and/or`, this should require both operands to be boolean expressions (or explicitly disallow non-boolean `and/or` usage).

## Actual Behavior

- Any `ast.BoolOp` is classified as boolean, even when it returns non-boolean values per Python semantics.

## Evidence

- Boolean classifier treats all BoolOp as boolean: `src/elspeth/engine/expression_parser.py:426-430`
- Evaluator returns last truthy/falsy value (can be non-boolean): `src/elspeth/engine/expression_parser.py:284-297`
- Config validation relies on `is_boolean_expression()` to enforce route labels: `src/elspeth/core/config.py:276-303`

## Impact

- User-facing impact: valid configs are rejected or routes fail at runtime due to mismatched labels.
- Data integrity / security impact: low.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- The boolean classifier over-approximates by treating all `and/or` expressions as boolean, despite evaluation returning non-boolean values.

## Proposed Fix

- Code changes (modules/files):
  - Update `_is_boolean_node` to treat `ast.BoolOp` as boolean only if all operands are boolean expressions, or explicitly restrict `and/or` to boolean operands in validation.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests for expressions like `row.get('label') or 'unknown'` and `row['x'] and row['y']` to ensure they are not classified as boolean.
- Risks or migration steps:
  - Existing configs that rely on `and/or` for boolean results should continue to pass; document the stricter classification.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (conditions can return route labels)
- Observed divergence: classifier forces boolean routing for expressions that return labels.
- Reason (if known): simplified static check.
- Alignment plan or decision needed: clarify intended semantics for `and/or` in gate conditions.

## Acceptance Criteria

- `is_boolean_expression()` accurately reflects whether evaluation returns a boolean for `and/or` expressions.

## Tests

- Suggested tests to run: `pytest tests/engine/test_expression_parser.py -k boolean_expression`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
