# Bug Report: User-defined sink names are lowercased during config load

## Summary

- `_lowercase_schema_keys()` lowercases sink names (dict keys in `sinks` section) which can break `default_sink` references using mixed case.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/config.py:1311-1336` - `_lowercase_schema_keys()` lowercases all schema-level dict keys (including sink names) except inside `options`.
- Sink names like `MySink` become `mysink`
- `default_sink: MySink` would then fail validation

## Impact

- User-facing impact: Mixed-case sink names break unexpectedly
- Data integrity: None

## Proposed Fix

- Preserve sink name casing or document lowercase-only requirement

## Acceptance Criteria

- Sink names preserve original casing, or validation fails early with clear message

## Verification (2026-02-01)

**Status: STILL VALID**

- `_lowercase_schema_keys()` still lowercases sink names at the top level. (`src/elspeth/core/config.py:1311-1336`)

---

## Resolution (2026-02-02)

**Status: CLOSED - FIXED**

### Design Decision

Rather than silently transforming sink names, we chose the explicit approach:

1. **Preserve sink name casing** - `_lowercase_schema_keys()` now treats `sinks` like `options`, preserving user-defined keys
2. **Enforce lowercase at validation** - A new validator rejects non-lowercase sink names with a helpful error message

This follows the codebase philosophy: no silent transformations, fail fast with clear errors.

### Fix Applied

1. Updated `_lowercase_schema_keys()` to add `sinks` to the preserve list (like `options`)
2. Added `validate_sink_names_lowercase()` field validator that rejects mixed-case sink names

**Example error message:**
```
Sink names must be lowercase. Found: ['MyOutput'].
Suggested fixes: 'MyOutput' -> 'myoutput'
```

### Files Changed

- `src/elspeth/core/config.py:1333-1365` - Updated `_lowercase_schema_keys()` to preserve `sinks` keys
- `src/elspeth/core/config.py:895-912` - Added `validate_sink_names_lowercase()` validator

### Tests Added

- `tests/core/test_config.py::TestSinkNameCasing` - 6 new tests covering:
  - Lowercase sink names accepted
  - Mixed-case sink names rejected with helpful error
  - Uppercase sink names rejected
  - Underscores and numbers in lowercase names work
  - `default_sink` must reference existing sink
  - YAML loading preserves sink names (no silent lowercasing)
