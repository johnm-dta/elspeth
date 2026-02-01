# Bug Report: Trigger condition silently coerces non-boolean results

## Summary

- Trigger evaluation uses `if bool(result):` which coerces non-boolean expression results. Integer or string expressions trigger flushes based on truthiness rather than explicit boolean.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/engine/triggers.py:125-136` - Line 134: `if bool(result):` coerces non-boolean
- A condition like `"row['batch_count']"` (integer) would trigger on any non-zero value
- Violates CLAUDE.md prohibition on coercion for our data

## Impact

- User-facing impact: Unexpected batch boundaries from truthy non-boolean values
- Data integrity: Aggregation behavior depends on Python truthiness rules

## Proposed Fix

- Validate that expression result is actually boolean, raise if not

## Acceptance Criteria

- Non-boolean expression results raise ValueError
- Boolean results work as expected

## Verification (2026-02-01)

**Status: STILL VALID**

- Trigger condition still coerces result with `bool(result)` instead of enforcing boolean. (`src/elspeth/engine/triggers.py:125-135`)

## Resolution (2026-02-02)

**Status: FIXED**

**Fix implemented in two layers (defense-in-depth):**

1. **Config-time validation** (`src/elspeth/core/config.py:84-113`):
   - `TriggerConfig.validate_condition_expression()` now uses `ExpressionParser.is_boolean_expression()` to reject non-boolean expressions at config load time
   - Clear error message guides users to use comparisons or boolean operators

2. **Runtime validation** (`src/elspeth/engine/triggers.py:124-132, 173-181`):
   - Both `record_accept()` and `should_trigger()` now validate `isinstance(result, bool)` before using the condition result
   - Raises `TypeError` with expression details if non-boolean detected
   - Defense-in-depth in case config validation is bypassed

**Tests added:**
- `tests/engine/test_triggers.py::TestTriggerConditionBooleanValidation` (4 tests)
  - `test_non_boolean_condition_rejected_at_config_time`
  - `test_boolean_condition_accepted`
  - `test_ternary_with_boolean_branches_accepted`
  - `test_non_boolean_runtime_raises`
