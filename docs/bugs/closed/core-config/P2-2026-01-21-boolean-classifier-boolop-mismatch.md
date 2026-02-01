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

## Verification (2026-01-25)

**Status: STILL VALID**

### Code Review Findings

The bug persists in the current codebase at commit `d984cb4` on branch `fix/rc1-bug-burndown-session-4`:

1. **Boolean Classifier (lines 426-430 in expression_parser.py)**:
   ```python
   # Boolean operators (and, or) always return truthy/falsy value
   # Note: In Python, `x and y` returns y if x is truthy, not necessarily bool
   # But for gate routing purposes, we treat this as boolean-ish
   if isinstance(node, ast.BoolOp):
       return True
   ```
   The code contains a comment acknowledging that `and/or` don't necessarily return booleans, but dismisses this as "boolean-ish" for gate routing. This is the root cause.

2. **Evaluator Returns Non-Boolean Values (lines 284-297)**:
   The `visit_BoolOp` method correctly implements Python semantics - returning the last truthy/falsy value, which can be any type (string, number, etc.), not just bool.

3. **Config Validation Enforces true/false (lines 276-294 in config.py)**:
   When `is_boolean_expression()` returns True, validation requires route labels to be exactly `{"true": ..., "false": ...}`, rejecting valid configs with semantic labels.

### Behavioral Verification

Tested the exact scenario from the bug report:

```python
# Expression: row.get('label') or 'unknown'
# is_boolean_expression(): True  ← WRONG, should be False
# Evaluate with label="vip": 'vip' (type: str)  ← Returns string, not bool
# Evaluate with missing label: 'unknown' (type: str)  ← Returns string, not bool

# Expression: row['x'] and row['y']
# is_boolean_expression(): True  ← WRONG, should be False
# Evaluate with x="hello", y="world": 'world' (type: str)  ← Returns string, not bool
```

### Architecture Contract Confirmation

`src/elspeth/contracts/routing.py:98` explicitly documents that gates can return semantic route labels:
```python
"""Route to a specific labeled destination.

Gates return semantic route labels (e.g., "above", "below", "match").
The executor resolves these labels via the plugin's `routes` config
to determine the actual destination (sink name or "continue").
```

This confirms the architectural intent supports non-boolean route labels, making the boolean classifier's behavior incorrect.

### Test Coverage Gap

Existing tests in `tests/engine/test_expression_parser.py:959-968` incorrectly assert that boolean operators ARE boolean:
```python
def test_boolean_operators_are_boolean(self) -> None:
    """Boolean operators (and, or) return boolean."""
    expressions = [
        "row['x'] > 0 and row['y'] > 0",  # This IS boolean (both operands boolean)
        "row['x'] == 1 or row['y'] == 2",  # This IS boolean (both operands boolean)
        ...
    ]
```

The test only covers cases where operands are comparisons (which ARE boolean). Missing test cases for non-boolean operands like `row.get('label') or 'unknown'`.

### Git History

No fixes found since the bug was reported on 2026-01-21. The expression_parser.py file has only seen formatting changes:
- `c786410`: RC-1 release
- `07084c3`: delint and reformat

### Impact Assessment

This bug affects any gate configuration using `and/or` for non-boolean value selection:
- **Config Rejection**: Valid configs like `{"condition": "row.get('status') or 'pending'", "routes": {"pending": "review", "approved": "continue"}}` are rejected
- **Runtime Mismatch**: If users work around validation by using `{"true": ..., "false": ...}` routes, runtime will fail because expression returns strings, not "true"/"false"

### Recommended Priority

Maintain P2 - this blocks legitimate use cases for semantic routing with `and/or` operators, but workarounds exist (use ternary expressions or avoid `and/or` for non-boolean operands).
