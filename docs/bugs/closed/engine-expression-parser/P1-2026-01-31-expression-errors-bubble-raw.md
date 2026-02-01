# Bug Report: Expression evaluation errors bubble as raw exceptions

## Summary

- Expression parser evaluation errors (KeyError, ZeroDivisionError, TypeError) bubble up as raw exceptions instead of a dedicated `ExpressionEvaluationError`, causing gate expressions to crash runs with opaque errors.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/engine/expression_parser.py:305-309` - `visit_Subscript()` returns `value[key]` directly and can raise `KeyError`.
- `src/elspeth/engine/expression_parser.py:353-358` - `visit_BinOp()` returns `op_func(left, right)` directly and can raise `ZeroDivisionError` / `TypeError`.
- `src/elspeth/engine/expression_parser.py:503-513` - `evaluate()` calls `evaluator.visit()` without wrapping or re-raising as a dedicated evaluation error.
- Design doc `docs/plans/completed/plugin-refactor/2026-01-18-wp09-engine-level-gates.md` specifies `ExpressionEvaluationError` which does not exist

## Impact

- User-facing impact: Gate expressions crash with confusing KeyError/ZeroDivisionError instead of clear message
- Data integrity / security impact: None (fails safe)
- Performance or cost impact: Developer time debugging opaque errors

## Root Cause Hypothesis

- Missing exception wrapping in expression evaluation - errors should be caught and re-raised as `ExpressionEvaluationError` with context.

## Proposed Fix

- Code changes:
  - Create `ExpressionEvaluationError` exception class
  - Wrap evaluation in try/except, catch common errors, re-raise with expression context
- Tests to add/update:
  - Add test with missing field access, assert `ExpressionEvaluationError` raised
  - Add test with division by zero, assert `ExpressionEvaluationError` raised

## Acceptance Criteria

- Expression evaluation errors raise `ExpressionEvaluationError` with expression text and original error
- Error messages are actionable (include field name, expression, row context)

## Resolution

**Status: FIXED** (2026-02-01)

### Changes Made

1. **Added `ExpressionEvaluationError` exception class** (lines 32-42)
   - Wraps operational errors with context
   - Chains original exception via `__cause__` for debugging

2. **Updated `visit_Subscript()` to wrap KeyError and TypeError** (lines 317-335)
   - Provides helpful error message with field name
   - Lists available fields when accessing missing key

3. **Updated `visit_BinOp()` to wrap ZeroDivisionError and TypeError** (lines 383-398)
   - Catches division by zero in `/`, `//`, `%` operations
   - Catches type errors in arithmetic operations

4. **Updated `visit_Compare()` to wrap TypeError** (lines 351-367)
   - Catches type errors when comparing incompatible types

5. **Updated `visit_Set()` and `visit_Dict()` to wrap TypeError** (lines 410-432)
   - Handles unhashable type errors in literal construction

6. **Updated `visit_Subscript()` to also wrap IndexError** (lines 318-338)
   - Catches out-of-range index access on lists/tuples
   - Error message includes index and collection length

7. **Updated `visit_UnaryOp()` to wrap TypeError** (lines 406-414)
   - Catches type errors in unary operations (e.g., `-'hello'`)

8. **Updated `visit_Call()` to wrap TypeError** (lines 349-357)
   - Catches unhashable key errors in `row.get()` calls (e.g., `row.get([])`)

9. **Fixed KeyError cause chain** (line 324)
   - Changed `from None` to `from e` to preserve original KeyError as `__cause__`

### Tests Added

- `TestExpressionEvaluationError` class with 19 new tests:
  - `test_missing_field_raises_evaluation_error`
  - `test_missing_field_error_includes_available_fields`
  - `test_division_by_zero_raises_evaluation_error`
  - `test_floor_division_by_zero_raises_evaluation_error`
  - `test_modulo_by_zero_raises_evaluation_error`
  - `test_type_error_raises_evaluation_error`
  - `test_nested_field_missing_raises_evaluation_error`
  - `test_evaluation_error_includes_expression_text`
  - `test_evaluation_error_preserves_original_exception`
  - `test_comparison_type_error_raises_evaluation_error`
  - `test_index_out_of_range_raises_evaluation_error`
  - `test_index_error_includes_length_info`
  - `test_unary_minus_type_error_raises_evaluation_error`
  - `test_unary_not_on_any_type_succeeds`
  - `test_negative_index_out_of_range_raises_evaluation_error`
  - `test_unary_plus_type_error_raises_evaluation_error`
  - `test_tuple_index_out_of_range_raises_evaluation_error`
  - `test_row_get_unhashable_key_raises_evaluation_error`
  - `test_missing_field_error_preserves_cause`

### Verification

All 142 expression parser tests pass, including:
- 19 new `ExpressionEvaluationError` tests
- All existing fuzz tests (updated to suppress new exception type)
- All gate-related integration tests
