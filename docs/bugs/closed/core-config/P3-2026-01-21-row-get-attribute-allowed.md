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

## Verification (2026-01-25)

**Status: STILL VALID**

### Verification Method

1. Examined current codebase at commit `7540e57` on branch `fix/rc1-bug-burndown-session-4`
2. Reviewed `src/elspeth/engine/expression_parser.py` implementation
3. Checked git history for any fixes since bug was reported (2026-01-21)
4. Created and executed test cases to verify actual behavior

### Findings

The bug is **still present** in the current codebase. The `visit_Attribute` method (lines 90-97) allows `row.get` attribute access without verifying that it's part of a `Call` node.

**Current behavior (lines 90-97):**
```python
def visit_Attribute(self, node: ast.Attribute) -> None:
    """Allow only row.get method access."""
    if isinstance(node.value, ast.Name) and node.value.id == "row":
        if node.attr != "get":
            self.errors.append(f"Forbidden row attribute: {node.attr!r} (only 'get' is allowed)")
    else:
        self.errors.append(f"Forbidden attribute access: {node.attr!r}")
    self.generic_visit(node)
```

**Problem:** The method allows `row.get` attribute access unconditionally. It only checks that:
1. The value is the name `row`
2. The attribute is `get`

It does NOT check whether the attribute access is followed by a call. This means bare `row.get` passes validation.

### Test Results

Created test cases to verify the issue:

```python
# Test 1: Bare row.get (should be rejected but is allowed)
parser = ExpressionParser("row.get")
# Result: ALLOWED - evaluates to <built-in method get of dict object>

# Test 2: row.get() with args (correctly allowed)
parser = ExpressionParser("row.get('field')")
# Result: ALLOWED - evaluates correctly to field value

# Test 3: row.get in comparison (should be rejected but is allowed)
parser = ExpressionParser("row.get == None")
# Result: ALLOWED - evaluates to method object comparison
```

### Impact Assessment

**Severity remains P3 (minor)** because:
- No evidence found of this pattern in production configs
- Would fail at runtime with a clear error (route label would be a method object)
- Primarily a config validation gap, not a security or data integrity issue

### Git History

No fixes found in git history since bug was reported:
- No commits to `expression_parser.py` since RC1 (commit `c786410`)
- No related fixes in the RC1 bug fix batch (commit `57c57f5`)
- Expression parser was added in commit `3e1a127` and hasn't been modified for this issue

### Root Cause Confirmed

The `visit_Attribute` method validates attribute access in isolation, without checking the parent AST node context. To fix this, the validator needs to:

1. Track parent node context to determine if an `Attribute` node is the `func` of a `Call` node, OR
2. Only validate `row.get` within `visit_Call`, and reject all `Attribute` nodes with `row.get` in `visit_Attribute`

### Recommendation

**Defer to post-RC1** - This is a config validation quality issue that:
- Has no impact on running pipelines (no configs use this pattern)
- Would be caught immediately at runtime if misconfigured
- Does not affect security or audit integrity
- Can be addressed in a future enhancement pass

The fix should include:
- Rejection of bare `row.get` attribute access
- Test case: `pytest.raises(ExpressionSecurityError, match="...", ExpressionParser("row.get"))`
- Test case: `pytest.raises(ExpressionSecurityError, match="...", ExpressionParser("row.get == None"))`
