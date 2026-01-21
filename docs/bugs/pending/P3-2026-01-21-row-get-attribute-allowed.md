# Bug Report: ExpressionParser allows bare `row.get` attribute access

## Summary

- The validator permits `row.get` attribute access even when it is not called. This results in the expression evaluating to a bound method object, which is not a meaningful route label and can cause runtime routing failures.

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
- Notable tool calls or steps: code inspection of attribute handling

## Steps To Reproduce

1. Configure a gate condition `row.get`.
2. `ExpressionParser` accepts the expression.
3. At runtime, `evaluate()` returns a bound method object, which becomes the route label and fails route lookup.

## Expected Behavior

- Attribute access should only be allowed when `row.get` is actually called (`row.get('field')`), not when used as a bare attribute.

## Actual Behavior

- `row.get` without a call is permitted and evaluates to a method object.

## Evidence

- Validator allows `row.get` attribute access without checking for call context: `src/elspeth/engine/expression_parser.py:90-97`
- Evaluator returns the bound method object for `row.get`: `src/elspeth/engine/expression_parser.py:259-264`

## Impact

- User-facing impact: misconfigured conditions pass validation but fail routing at runtime.
- Data integrity / security impact: low.
- Performance or cost impact: low.

## Root Cause Hypothesis

- Attribute validation allows `row.get` unconditionally rather than only in `row.get(...)` calls.

## Proposed Fix

- Code changes (modules/files):
  - Disallow bare `row.get` attribute access by tightening `visit_Attribute`, or explicitly reject `ast.Attribute` unless it is part of a `Call` node.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that rejects `row.get` without arguments.
- Risks or migration steps:
  - Existing configs using `row.get` as a value must be updated (likely mistakes).

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (allowed access is `row.get('field')`)
- Observed divergence: attribute access without call is permitted.
- Reason (if known): attribute validation is not coupled to call context.
- Alignment plan or decision needed: enforce row.get usage only as a call.

## Acceptance Criteria

- Expressions containing bare `row.get` are rejected at validation time.

## Tests

- Suggested tests to run: `pytest tests/engine/test_expression_parser.py -k row_get`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
