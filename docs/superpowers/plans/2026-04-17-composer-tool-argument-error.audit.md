# Composer tools.py raise-site audit

Ran by: Task 1.5 Step 1 AST grep (date: 2026-04-17).

Scope: `src/elspeth/web/composer/tools.py` — every `raise TypeError|ValueError|UnicodeError|UnicodeDecodeError|UnicodeEncodeError` node, classified by intent.

## Findings (classify each)

| Line | Class     | Classification | Notes |
|------|-----------|----------------|-------|
| 1753 | TypeError | Convert        | Tier-3 content-type guard for `_execute_create_blob`; Task 2 converts to `ToolArgumentError`. |
| 1838 | TypeError | Convert        | Tier-3 content-type guard for `_execute_update_blob`; Task 3 converts to `ToolArgumentError`. |

No other `raise TypeError`, `raise ValueError`, `raise UnicodeError`, `raise UnicodeDecodeError`, or `raise UnicodeEncodeError` nodes exist in `tools.py` as of this commit.

## Implicit coercion scan

Grepped for `int(arguments[`, `float(arguments[`, `tuple(arguments[`, `list(arguments[` in handler bodies. **No matches.** No implicit Tier-3 coercion sites require wrapping.

## Classifications

- **Convert** — deliberate Tier-3 guard raising the wrong class. Target for Task 2/3/5 conversion.
- **Internal invariant** — plugin-bug signal that SHOULD propagate. Leave as-is; narrowed catch lets it through.
- **Locally contained** — inside `try/except ... return _failure_result(...)`. Leave as-is; never escapes handler.

## Task 7 allowlist seeds

Every row classified as "Internal invariant" becomes an allowlist entry in
`config/cicd/enforce_composer_exception_channel/_defaults.yaml`. Every row
classified as "Convert" becomes a Task 2/3/5 conversion item. Every row
classified as "Locally contained" requires no action but should be documented
here for the next audit.

**Post-conversion state (after Tasks 2 and 3 land):** both "Convert" rows are resolved. Zero "Internal invariant" rows remain. **Task 7 allowlist is empty** — `allowed: []`.
