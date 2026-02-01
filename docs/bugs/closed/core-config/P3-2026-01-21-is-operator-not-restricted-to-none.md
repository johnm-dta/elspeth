# Bug Report: ExpressionParser allows `is`/`is not` comparisons with non-None operands

## Summary

- The parser allows `is`/`is not` comparisons against any value, even though documentation states identity checks are intended for `None` only. This can introduce subtle logic bugs when users write `row['x'] is 1` (identity) instead of equality.

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
- Notable tool calls or steps: code inspection of compare operator validation

## Steps To Reproduce

1. Use a condition like `row['x'] is 1` or `row['x'] is not 'foo'`.
2. `ExpressionParser` accepts the expression.
3. Evaluation uses identity semantics, producing surprising results.

## Expected Behavior

- `is`/`is not` should be restricted to `None` checks (e.g., `row.get('x') is None`) per documentation, rejecting other identity comparisons at validation time.

## Actual Behavior

- Any `is`/`is not` comparison is permitted.

## Evidence

- `ast.Is` and `ast.IsNot` are allowed without operand restrictions: `src/elspeth/engine/expression_parser.py:40-43`, `src/elspeth/engine/expression_parser.py:117-122`
- Documentation frames identity comparisons as None checks: `src/elspeth/engine/expression_parser.py:351-353`

## Impact

- User-facing impact: subtle logic errors if users mistakenly write identity checks for non-None values.
- Data integrity / security impact: low.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Validation does not inspect operands for `is`/`is not` to ensure `None` usage.

## Proposed Fix

- Code changes (modules/files):
  - In `_ExpressionValidator.visit_Compare`, when encountering `ast.Is`/`ast.IsNot`, enforce that one side is a `None` literal.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests that reject `row['x'] is 1` and accept `row.get('x') is None`.
- Risks or migration steps:
  - Existing configs using `is` for non-None comparisons must be updated to `==`.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` and `src/elspeth/engine/expression_parser.py` (identity checks noted for None)
- Observed divergence: identity comparisons are unrestricted.
- Reason (if known): missing operand validation.
- Alignment plan or decision needed: clarify whether identity comparisons beyond None should ever be allowed.

## Acceptance Criteria

- `is`/`is not` comparisons are only accepted when checking against `None`.

## Tests

- Suggested tests to run: `pytest tests/engine/test_expression_parser.py -k is_none`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Verification Steps

1. **Examined current codebase** (commit 7540e57 on branch `fix/rc1-bug-burndown-session-4`)
   - File: `/home/john/elspeth-rapid/src/elspeth/engine/expression_parser.py`
   - The `_ExpressionValidator.visit_Compare` method (lines 117-122) validates that comparison operators are in the allowed list but does NOT check operands
   - `ast.Is` and `ast.IsNot` are in `_COMPARISON_OPS` (lines 40-41) without restrictions

2. **Checked documentation**
   - Line 352: Documentation explicitly states `"Identity: is, is not (for None checks)"`
   - This confirms the intended behavior is to restrict `is`/`is not` to None comparisons only

3. **Tested current behavior**
   ```python
   # ALLOWED (but shouldn't be according to docs):
   ExpressionParser("row['x'] is 1")  # Accepts and evaluates with identity semantics
   ExpressionParser("row['status'] is 'active'")  # Accepts (but evaluates to False due to string interning)

   # ALLOWED (correct behavior):
   ExpressionParser("row.get('x') is None")  # Accepts and works correctly
   ```

4. **Checked git history**
   - No commits found that address this issue
   - Expression parser introduced in commit 3e1a127 (2026-01-18) without operand validation for `is`/`is not`
   - Bug fix commit 57c57f5 (2026-01-21) fixed 8 other RC1 bugs but not this one
   - No changes to expression_parser.py since RC1 except formatting (commit 07084c3)

### Root Cause Confirmed

The `visit_Compare` method in `_ExpressionValidator` only validates that the operator type is allowed, but does not examine the operands (left, comparators) to ensure `is`/`is not` are only used with `None` literals.

Current code (lines 117-122):
```python
def visit_Compare(self, node: ast.Compare) -> None:
    """Validate comparison operators."""
    for op in node.ops:
        if type(op) not in _COMPARISON_OPS:
            self.errors.append(f"Forbidden comparison operator: {type(op).__name__}")
    self.generic_visit(node)
```

This needs additional logic to check if `op` is `ast.Is` or `ast.IsNot`, and if so, validate that at least one operand is a `ast.Constant` with value `None`.

### Impact Assessment

- **Risk**: Low-Medium. Users could write expressions like `row['x'] is 1` thinking it's equality, but Python's identity semantics would apply
- **Likelihood**: Medium. The documentation suggests `is` is for None checks, so users might assume this restriction is enforced
- **Data Integrity**: Low. This would cause logic errors in gate routing, not silent data corruption
- **Example failure scenario**: `row['count'] is 1` would evaluate to `True` for the specific integer object `1` in CPython's small integer cache (-5 to 256), but `False` for `row['count'] = int('1')` created dynamically

### Recommendation

This bug should remain **P3** priority as it:
1. Does not cause data integrity issues (just routing logic errors)
2. Is unlikely to occur in practice if users follow documentation
3. Would be caught during testing when gate routing behaves unexpectedly
4. Has a clear fix path that can be implemented when time permits

However, it should be fixed before GA release to prevent user confusion and ensure documentation matches implementation.
