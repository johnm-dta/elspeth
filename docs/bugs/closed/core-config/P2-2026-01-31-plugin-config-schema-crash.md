# Bug Report: PluginConfig.from_dict crashes on schema: null or non-mapping configs

## Summary

- `PluginConfig.from_dict()` calls `SchemaConfig.from_dict(schema_dict)` without checking if `schema_dict` is a mapping. TypeError not caught by `except ValueError`.

## Severity

- Severity: moderate (P3 original, upgraded)
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/config_base.py:59-64` - calls `SchemaConfig.from_dict(schema_dict)` without checking type.
- `src/elspeth/contracts/schema.py:280-286` - `from_dict()` assumes `config` is a mapping; `None` triggers `TypeError` on `"fields" not in config`.
- `src/elspeth/plugins/config_base.py:65-68` - only catches `ValidationError` / `ValueError`, not `TypeError`.

## Impact

- User-facing impact: Confusing TypeError on malformed config
- Data integrity: None (fails fast)

## Proposed Fix

- Check `isinstance(schema_dict, dict)` before calling `SchemaConfig.from_dict()`

## Acceptance Criteria

- Clear error message for `schema: null` or non-dict schema values

## Verification (2026-02-01)

**Status: STILL VALID**

- `PluginConfig.from_dict()` still passes `schema` directly to `SchemaConfig.from_dict()` without type checks; `TypeError` is not caught. (`src/elspeth/plugins/config_base.py:59-68`, `src/elspeth/contracts/schema.py:280-286`)

---

## Resolution (2026-02-02)

**Status: CLOSED - FIXED**

### Fix Applied

Added type guard in `PluginConfig.from_dict()` before calling `SchemaConfig.from_dict()`:

```python
if not isinstance(schema_dict, dict):
    raise PluginConfigError(
        f"Invalid configuration for {cls.__name__}: "
        f"'schema' must be a dict, got {type(schema_dict).__name__}. "
        f"Use 'schema: {{fields: dynamic}}' or provide explicit field definitions."
    )
```

### Files Changed

- `src/elspeth/plugins/config_base.py:62-68` - Added type check with clear error message

### Tests Added

- `tests/core/test_config.py::TestPluginConfigSchemaValidation` - 5 new tests covering:
  - `schema: null` → clear error
  - `schema: "string"` → clear error
  - `schema: [list]` → clear error
  - `schema: 123` → clear error
  - Valid schema dict → works
